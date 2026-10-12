"""Causal implicit equilibrium of each constituent's rigid motions.

The neural field supplies deformation. Eighteen motion coordinates close the
three bodies' force/moment equations at each material input and physical step.
Only surface forces enter this solve; bulk strain and the reference are kept.
The backward pass differentiates the converged equations and cohesive history.
"""

import numpy as np
import torch


class ConstituentReactionBalance:
    def __init__(self, problem):
        p = self.physics = problem.physics
        self.compliance = problem.config["apparatus"]["compliance_m_per_n"]
        self.reference_force = problem.reference_force
        self.machine = problem.machine
        xyz = p.mesh["xyz"]
        centre = np.array([0., 0., xyz[:, 2].min()])
        fields = np.zeros((len(xyz), 3, 18))
        for body in range(3):
            nodes = np.unique(p.mesh["tetra"][p.mesh["body"] == body])
            fields[nodes, :, 6*body:6*body+3] = np.eye(3)
            for axis in range(3):
                fields[nodes, :, 6*body+3+axis] = np.cross(np.eye(3)[axis], xyz[nodes]-centre)
        modes = p.tensor(fields)
        self.modes = modes.reshape(p.ndof, 18)
        self.jump_modes = torch.einsum("qa,fack->fqck", p.shape,
            modes[p.upper]-modes[p.lower]).reshape(-1, 3, 18)
        self.wall_modes = torch.einsum("qa,fack,fc->fqk", p.shape,
            modes[p.wall_substrate]-modes[p.wall_cone], p.wall_normal).reshape(-1, 18)
        self.support_modes = torch.einsum("qa,fak->fqk", p.shape,
            modes[p.trestle_faces, 2]).reshape(-1, 18)
        self.guide = p.tensor(p.mesh["lateral_guide_operator"])
        self.guide_modes = self.guide@self.modes
        self.anchor_modes = p.anchor_operator@self.modes
        self.jack_modes = p.jack_operator@self.modes
        self.gravity = p.gravity@self.modes
        self.area = (p.area[:, None]*p.surface_weights).flatten()
        geometry = p.settings["geometry"]
        self.wall_stiffness = (p.wall_area[:, None]*p.surface_weights
            *geometry["cone_contact_stiffness_pa_per_m"]).flatten()
        self.support_stiffness = (p.trestle_face_area[:, None]*p.surface_weights
            *geometry["trestle_stiffness_pa_per_m"]).flatten()
        self.guide_stiffness = geometry["lateral_guide_stiffness_n_per_m"]
        # An engaged elastic metric only scales the 18 unknowns. Equations and
        # the final implicit tangent always use the actual contact/damage state.
        metric = (p.interface_stiffness*torch.einsum("qci,qcj,q->ij",
            self.jump_modes, self.jump_modes, self.area)
            +self.wall_modes.T@(self.wall_stiffness[:, None]*self.wall_modes)
            +self.support_modes.T@(self.support_stiffness[:, None]*self.support_modes)
            +self.anchor_modes.T@(p.anchor_stiffness[:, None]*self.anchor_modes)
            +self.guide_stiffness*self.guide_modes.T@self.guide_modes
            +torch.outer(self.jack_modes, self.jack_modes)/self.compliance)
        root = torch.linalg.cholesky(metric)
        transform = torch.linalg.solve_triangular(root.T, torch.eye(18, dtype=p.dtype), upper=True)
        for name in ("modes", "jump_modes", "wall_modes", "support_modes", "guide_modes", "anchor_modes"):
            setattr(self, name, getattr(self, name)@transform)
        self.jack_modes = self.jack_modes@transform
        self.gravity = self.gravity@transform
        self.sizes = [self.jump_modes.shape[0]*3, self.wall_modes.shape[0],
                      self.anchor_modes.shape[0], self.support_modes.shape[0],
                      self.guide_modes.shape[0], 1]

    def boundary_values(self, displacement):
        p = self.physics
        nodal = displacement.reshape(*displacement.shape[:-1], -1, 3)
        jump = torch.einsum("qa,...fac->...fqc", p.shape, nodal[..., p.upper, :]-nodal[..., p.lower, :])
        support = torch.einsum("qa,...fa->...fq", p.shape, nodal[..., p.trestle_faces, 2])
        rod = displacement@p.anchor_operator.T-p.anchor_reference+p.assembly_tension/p.anchor_stiffness
        return torch.cat([jump.flatten(-3), p.wall_gap(displacement).flatten(-2), rod,
            support.flatten(-2), displacement@self.guide.T, (displacement@p.jack_operator)[..., None]], dim=-1)

    def effective(self, jump):
        positive = torch.cat([jump[..., :2], jump[..., 2:].clamp_min(0)], dim=-1)
        return positive.square().sum(-1).clamp_min((self.physics.onset*1e-12)**2).sqrt()

    def equations(self, correction, offsets, prior, compression, target, *, tangent=False):
        p = self.physics
        jump0, wall0, rod0, support0, guide0, jack0 = offsets.split(self.sizes, dim=-1)
        jump = jump0.reshape(*jump0.shape[:-1], -1, 3)+torch.einsum("...k,qck->...qc", correction, self.jump_modes)
        wall = wall0+correction@self.wall_modes.T
        rod = rod0+correction@self.anchor_modes.T
        support = support0+correction@self.support_modes.T
        guide = guide0+correction@self.guide_modes.T
        jack = jack0[..., 0]+correction@self.jack_modes-target
        traction, _, history = p.interface_step(jump, prior, compression[..., None])
        residual = (torch.einsum("...qc,qck,q->...k", traction, self.jump_modes, self.area)
            +(wall.clamp_max(0)*self.wall_stiffness)@self.wall_modes
            +(rod.clamp_min(0)*p.anchor_stiffness)@self.anchor_modes
            +(support.clamp_max(0)*self.support_stiffness)@self.support_modes
            +self.guide_stiffness*guide@self.guide_modes
            +jack[..., None]*self.jack_modes/self.compliance-self.gravity)
        if not tangent:
            return residual, history
        # Each quadrature point is independent at fixed prior history. Three
        # pullbacks give its 3x3 tangent, including autograd's selected branch
        # at equal maxima and at damage clipping boundaries.
        with torch.enable_grad():
            point = jump.detach().requires_grad_(True)
            current = p.interface_step(point, prior.detach(), compression.detach()[..., None])[0]
            local = torch.stack([torch.autograd.grad(current[..., axis].sum(), point,
                retain_graph=axis < 2)[0] for axis in range(3)], dim=-2)
        matrix = (torch.einsum("qci,...qcd,qdj,q->...ij", self.jump_modes, local, self.jump_modes, self.area)
            +torch.einsum("...q,qi,qj->...ij", (wall < 0)*self.wall_stiffness, self.wall_modes, self.wall_modes)
            +torch.einsum("...q,qi,qj->...ij", (rod > 0)*p.anchor_stiffness, self.anchor_modes, self.anchor_modes)
            +torch.einsum("...q,qi,qj->...ij", (support < 0)*self.support_stiffness,
                          self.support_modes, self.support_modes)
            +self.guide_stiffness*self.guide_modes.T@self.guide_modes
            +torch.outer(self.jack_modes, self.jack_modes)/self.compliance)
        return residual, history, matrix

    @torch.no_grad()
    def solve(self, offsets, prior, compression, target, *, observe=None):
        correction = offsets.new_zeros((*offsets.shape[:-1], 18))
        tolerance = 1e-10
        for _ in range(80):
            residual, history, tangent = self.equations(correction, offsets, prior, compression, target, tangent=True)
            if observe is not None:
                observe(correction, residual, history, tangent)
            if float(residual.abs().max()) < tolerance:
                return correction, history, tangent
            direction = torch.linalg.solve(tangent, -residual[..., None])[..., 0]
            objective = residual.square().sum(-1)
            alpha = torch.ones_like(objective)
            for _ in range(50):
                trial = correction+alpha[..., None]*direction
                value, _ = self.equations(trial, offsets, prior, compression, target)
                acceptable = ((value.square().sum(-1) <= (1-1e-4*alpha)*objective)
                              | (value.abs().amax(-1) < tolerance))
                if bool(acceptable.all()):
                    correction = trial
                    break
                alpha = torch.where(acceptable, alpha, alpha*.5)
            else:
                raise RuntimeError("Constituent equilibrium line search failed")
        raise RuntimeError(f"Constituent equilibrium did not converge: {float(residual.abs().max())}")

    def __call__(self, displacement, compression):
        offsets = self.boundary_values(displacement)
        reference_jump = offsets[:, 0, :self.sizes[0]].reshape(len(displacement), -1, 3)
        history = self.physics.interface_path(reference_jump[:, None], compression[:, None, None])[2][:, -1]
        reference_jack = offsets[:, 0, -1]
        corrections = []
        for step in range(1, displacement.shape[1]):
            target = reference_jack+self.machine[step]+self.compliance*self.reference_force
            correction, history = ConstituentStep.apply(offsets[:, step], history, compression, target, self)
            corrections.append(correction)
        later = displacement[:, 1:]+torch.stack(corrections, dim=1)@self.modes.T
        return torch.cat([displacement[:, :1], later], dim=1)


class ConstituentStep(torch.autograd.Function):
    """First implicit derivative, including cached-primal JVPs via its dual.

    The active tangent is fixed in backward, as in the existing rigid layer;
    this implements the Gauss--Newton products, not physical Hessians.
    """

    @staticmethod
    def forward(ctx, offsets, prior, compression, target, balance):
        correction, history, tangent = balance.solve(offsets, prior, compression, target)
        ctx.balance = balance
        ctx.save_for_backward(offsets, prior, compression, target, correction, tangent)
        return correction, history

    @staticmethod
    def backward(ctx, correction_gradient, history_gradient):
        create_graph = torch.is_grad_enabled()
        offsets, prior, compression, target, correction, tangent = ctx.saved_tensors
        # Rebuild only the converged surface equations. Earlier-step gradients
        # pass through the returned history; Newton iterations are not unrolled.
        with torch.enable_grad():
            inputs = tuple(value.detach().requires_grad_(True) for value in (offsets, prior, compression, target))
            point = correction.detach().requires_grad_(True)
            residual, history = ctx.balance.equations(point, *inputs)
            history_pullback = torch.autograd.grad(history, point, history_gradient,
                retain_graph=True, create_graph=create_graph)[0]
            dual = torch.linalg.solve(tangent.transpose(-1, -2),
                (correction_gradient+history_pullback)[..., None])[..., 0]
            gradients = torch.autograd.grad((residual, history), inputs, (-dual, history_gradient),
                                            create_graph=create_graph)
        return (*gradients, None)
