"""Differentiable equilibrium of the six common rigid motions.

Only the reactive boundary and the series-compliance actuator do work on these
motions. Strain, cohesive separation and their histories remain unchanged.
The convex piecewise-quadratic boundary energy determines the rigid correction;
its active tangent supplies the implicit derivative used by neural training.
"""

import numpy as np
import torch


class RigidReactionBalance:
    def __init__(self, problem):
        p = self.physics = problem.physics
        self.compliance = problem.config["apparatus"]["compliance_m_per_n"]
        self.reference_force = problem.reference_force
        self.machine = problem.machine
        xyz = p.mesh["xyz"]
        modes = np.zeros((len(xyz), 3, 6))
        modes[:, :, :3] = np.eye(3)
        centre = np.array([0., 0., xyz[:, 2].min()])
        for axis in range(3):
            modes[:, :, 3+axis] = np.cross(np.eye(3)[axis], xyz-centre)
        self.modes = p.tensor(modes.reshape(p.ndof, 6))
        self.guide = p.tensor(p.mesh["lateral_guide_operator"])
        self.guide_stiffness = p.settings["geometry"]["lateral_guide_stiffness_n_per_m"]
        self.support_stiffness = (p.trestle_face_area[:, None]*p.surface_weights
            *p.settings["geometry"]["trestle_stiffness_pa_per_m"]).ravel()
        self.anchor_modes = p.anchor_operator@self.modes
        self.guide_modes = self.guide@self.modes
        self.support_modes = torch.einsum("qa,fak->fqk", p.shape,
            self.modes.reshape(-1, 3, 6)[p.trestle_faces, 2]).reshape(-1, 6)
        self.jack_modes = p.jack_operator@self.modes
        self.gravity = p.gravity@self.modes
        # Scale the small solve with its fully engaged boundary stiffness.
        metric = (self.guide_stiffness*self.guide_modes.T@self.guide_modes
            + self.anchor_modes.T@(p.anchor_stiffness[:, None]*self.anchor_modes)
            + self.support_modes.T@(self.support_stiffness[:, None]*self.support_modes)
            + torch.outer(self.jack_modes, self.jack_modes)/self.compliance)
        root = torch.linalg.cholesky(metric)
        transform = torch.linalg.solve_triangular(root.T, torch.eye(6, dtype=p.dtype), upper=True)
        for name in ("modes", "anchor_modes", "guide_modes", "support_modes"):
            setattr(self, name, getattr(self, name)@transform)
        self.jack_modes = self.jack_modes@transform
        self.gravity = self.gravity@transform

    def boundary_values(self, displacement):
        p = self.physics
        rod = (displacement@p.anchor_operator.T-p.anchor_reference
               +p.assembly_tension/p.anchor_stiffness)
        support = torch.einsum("qa,...fa->...fq", p.shape,
            displacement.reshape(*displacement.shape[:-1], -1, 3)[..., p.trestle_faces, 2])
        return rod, support.flatten(-2), displacement@self.guide.T, displacement@p.jack_operator

    def equations(self, correction, offsets, target):
        p = self.physics
        rod0, support0, guide0, jack0 = offsets
        rod = rod0+correction@self.anchor_modes.T
        support = support0+correction@self.support_modes.T
        guide = guide0+correction@self.guide_modes.T
        jack = jack0+correction@self.jack_modes-target
        rod_force = rod.clamp_min(0)*p.anchor_stiffness
        support_force = support.clamp_max(0)*self.support_stiffness
        gradient = (rod_force@self.anchor_modes+support_force@self.support_modes
            +self.guide_stiffness*guide@self.guide_modes
            +jack[..., None]*self.jack_modes/self.compliance-self.gravity)
        tangent = (torch.einsum("...a,ai,aj->...ij", (rod > 0)*p.anchor_stiffness,
                               self.anchor_modes, self.anchor_modes)
            +torch.einsum("...a,ai,aj->...ij", (support < 0)*self.support_stiffness,
                          self.support_modes, self.support_modes)
            +self.guide_stiffness*self.guide_modes.T@self.guide_modes
            +torch.outer(self.jack_modes, self.jack_modes)/self.compliance)
        energy = (.5*(rod.clamp_min(0)*rod_force).sum(-1)
            +.5*(support.clamp_max(0)*support_force).sum(-1)
            +.5*self.guide_stiffness*guide.square().sum(-1)
            +.5*jack.square()/self.compliance-correction@self.gravity)
        return gradient, tangent, energy, rod > 0, support < 0

    @torch.no_grad()
    def solve(self, displacement, reference):
        offsets = self.boundary_values(displacement)
        target = (reference@self.physics.jack_operator+self.machine[1:]
                  +self.compliance*self.reference_force)
        correction = displacement.new_zeros((*displacement.shape[:-1], 6))
        # This is algebraic solve accuracy, not a scientific acceptance target.
        tolerance = 1e-10
        for iteration in range(80):
            gradient, tangent, energy, rods, support = self.equations(correction, offsets, target)
            if float(gradient.abs().max()) < tolerance:
                return displacement+correction@self.modes.T, tangent, rods, support
            # With all rods/supports released the trial tangent can lack tilt
            # curvature. Regularization globalizes that trial only; the final
            # implicit derivative always uses the actual active tangent.
            shift = (1e-6-torch.linalg.eigvalsh(tangent)[..., 0]).clamp_min(0)
            step = torch.linalg.solve(tangent+shift[..., None, None]*torch.eye(6, dtype=displacement.dtype),
                                      -gradient[..., None])[..., 0]
            slope = (gradient*step).sum(-1)
            alpha = torch.ones_like(energy)
            for _ in range(50):
                trial = correction+alpha[..., None]*step
                trial_gradient, _, trial_energy, _, _ = self.equations(trial, offsets, target)
                acceptable = ((trial_energy <= energy+1e-4*alpha*slope
                               +32*torch.finfo(energy.dtype).eps*(1+energy.abs()))
                              | (trial_gradient.abs().amax(-1) < tolerance))
                if bool(acceptable.all()):
                    correction = trial
                    break
                alpha = torch.where(acceptable, alpha, alpha*.5)
            else:
                raise RuntimeError("Rigid boundary energy line search failed")
        raise RuntimeError(f"Rigid boundary equilibrium did not converge: {float(gradient.abs().max())}")

    def __call__(self, displacement):
        later = RigidBalanceFunction.apply(displacement[:, 1:], displacement[:, :1], self)
        return torch.cat([displacement[:, :1], later], dim=1)


class RigidBalanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, displacement, reference, balance):
        result, tangent, rods, support = balance.solve(displacement, reference)
        ctx.balance = balance
        ctx.save_for_backward(tangent, rods, support)
        return result

    @staticmethod
    def backward(ctx, gradient):
        b = ctx.balance
        p = b.physics
        tangent, rods, support = ctx.saved_tensors
        dual = torch.linalg.solve(tangent.transpose(-1, -2), (gradient@b.modes)[..., None])[..., 0]
        rod_dual = (dual@b.anchor_modes.T)*rods*p.anchor_stiffness
        support_dual = (dual@b.support_modes.T)*support*b.support_stiffness
        nodal = torch.zeros((*gradient.shape[:-1], p.ndof//3), dtype=gradient.dtype, device=gradient.device)
        local = torch.einsum("qa,...fq->...fa", p.shape,
            support_dual.reshape(*gradient.shape[:-1], *p.trestle_face_area.shape, -1))
        nodal.index_add_(-1, p.trestle_faces.flatten(), local.flatten(-2))
        correction = torch.zeros_like(gradient)
        correction[..., 2::3] = nodal
        correction = (correction+rod_dual@p.anchor_operator
            +b.guide_stiffness*(dual@b.guide_modes.T)@b.guide
            +(dual@b.jack_modes)[..., None]*p.jack_operator/b.compliance)
        reference_gradient = ((dual@b.jack_modes).sum(-1, keepdim=True)[..., None]
                              *p.jack_operator/b.compliance)
        return gradient-correction, reference_gradient, None
