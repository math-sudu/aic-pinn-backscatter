"""Current-state Newton equations in the PINN's existing spatial field.

Nonlocal damage derivatives remain in the matrix action; prior physical states
are held fixed. The existing reference-coordinate factors precondition the
equation. No measured response enters this solve, and no separate spatial
discretization is created.
"""

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, splu
import torch

from quadratic_elements import surface_matrix
from spatial_fracture import TensileSpectrum, elastic_stress, stress_components


def tensile_current_blocks(strain, prior, young, poisson, law, length, weights):
    """Split d sigma / d epsilon into local and nonlocal point blocks.

    Return C_local, d sigma / d averaged_strain and d equivalent / d strain.
    The history branch is the same torch.maximum branch as the physical path.
    """
    current = strain.detach().requires_grad_(True)
    effective = elastic_stress(current, current.new_tensor(young), poisson)
    positive, maximum = TensileSpectrum.apply(effective)
    equivalent = maximum.clamp_min(0)/young
    averaged = torch.sparse.mm(weights, equivalent[:, None])[:, 0]
    history = torch.maximum(prior, averaged)
    onset = law.tensile_strength_pa/young
    factor = torch.where(history > onset,
        law.strain_envelope(history, young, length)/(young*history.clamp_min(onset)),
        torch.ones_like(history))
    local = stress_components(effective+(factor.detach()-1)[:, None, None]*positive)
    elastic_part = torch.stack([torch.autograd.grad(local[:, i].sum(), current,
        retain_graph=True)[0] for i in range(6)], dim=1)
    gradient = torch.autograd.grad(equivalent.sum(), current, retain_graph=True)[0]
    rate = torch.autograd.grad(factor.sum(), averaged)[0]
    coupling = stress_components(positive.detach())*rate[:, None]
    return tuple(value.detach().numpy() for value in (elastic_part, coupling, gradient))


def body_current_blocks(fracture, displacement, young):
    """Assemble one body's local and nonlocal current-history tangent blocks."""
    p = fracture.physics
    path = displacement[None].detach()
    with torch.no_grad():
        strain_path = fracture.strains(path)[0]
        _, bulk = fracture.stress_path(path, p.tensor([young]))
        prior = (bulk["history"][0, -2] if len(displacement) > 1
                 else torch.zeros_like(bulk["history"][0, -1]))
    local, coupling, gradient = tensile_current_blocks(strain_path[-1], prior,
        young, p.settings["bulk"]["poisson"], fracture.law,
        fracture.softening_length, fracture.weights)
    b = fracture.b.numpy()
    npoint, width = len(fracture.points), b.shape[-1]
    dofs = (3*fracture.conn.numpy()[..., None]+np.arange(3)).reshape(-1, width)
    point_dofs = np.repeat(dofs, fracture.nquad, axis=0)
    strain_operator = sparse.coo_matrix((b.reshape(npoint, 6, width).ravel(),
        (np.broadcast_to(np.arange(npoint*6).reshape(npoint, 6, 1),
                         (npoint, 6, width)).ravel(),
         np.broadcast_to(point_dofs[:, None, :], (npoint, 6, width)).ravel())),
        shape=(6*npoint, p.ndof)).tocsr()
    rows = np.broadcast_to(np.arange(npoint)[:, None], (npoint, width)).ravel()
    equivalent = np.einsum("qi,qij->qj", gradient, b.reshape(npoint, 6, width))
    right = sparse.coo_matrix((equivalent.ravel(), (rows, point_dofs.ravel())),
                              shape=(npoint, p.ndof)).tocsr()
    point_coupling = sparse.coo_matrix(((coupling*fracture.volumes[:, None]).ravel(),
        (np.arange(6*npoint), np.repeat(np.arange(npoint), 6))),
        shape=(6*npoint, npoint)).tocsr()
    left = (strain_operator.T@point_coupling).tocsr()
    local = local.reshape(len(dofs), fracture.nquad, 6, 6)
    matrix = surface_matrix(fracture.conn.numpy(),
        np.einsum("eqai,eqab,eqbj,e->eij", b, local, b, fracture.measure.numpy(),
                  optimize=True), p.ndof)
    return matrix, left, fracture.averaging, right


class CurrentControlledTangent(LinearOperator):
    """Current-state tangent for actuator control or a prescribed reference load.

    The preceding bulk and interface histories are reconstructed from the
    supplied path. Rigid constituent corrections are left as actual unknowns
    in this full Newton equation; its converged fields satisfy those same
    eighteen force/moment equations. A prescribed load contributes no actuator
    stiffness; a single reference state starts with zero interface history.
    """

    def __init__(self, problem, displacement, eta, *, prescribed_force=False):
        p, fracture = problem.physics, problem.fracture
        self.coordinates = problem.coordinates
        material = problem.base.materials(eta[None])[0].detach().numpy()
        young, compression = map(float, material)
        path = displacement[None].detach()
        top_matrix, self.left, self.averaging, self.right = body_current_blocks(fracture, displacement, young)
        bulk_matrix = p._bulk_matrix-problem.base.top_stiffness_matrix+top_matrix
        if problem.substrate_fracture is not None:
            matrix, left, averaging, right = body_current_blocks(problem.substrate_fracture,
                displacement, p.settings["bulk"]["substrate_young_pa"])
            bulk_matrix = bulk_matrix-problem.substrate_stiffness_matrix+matrix
            self.left = sparse.hstack([self.left, left], format="csr")
            self.averaging = sparse.block_diag([self.averaging, averaging], format="csr")
            self.right = sparse.vstack([self.right, right], format="csr")

        # Surface tangent uses the actual updated maximum-separation history.
        with torch.no_grad():
            nodal = path.reshape(1, len(displacement), -1, 3)
            jumps = torch.einsum("qa,bsfac->bsfqc", p.shape,
                nodal[:, :, p.upper]-nodal[:, :, p.lower]).flatten(2, 3)[0]
            if len(displacement) > 1:
                prior_interface = p.interface_path(jumps[None, :-1], compression)[2][0, -1]
            else:
                shape = (jumps.shape[1], 3) if p.interface_friction_coefficient else (jumps.shape[1],)
                prior_interface = jumps.new_zeros(shape)
        jump = jumps[-1].detach().requires_grad_(True)
        traction = p.interface_step(jump, prior_interface, compression)[0]
        surface_tangent = torch.stack([torch.autograd.grad(traction[:, i].sum(), jump,
            retain_graph=i < 2)[0] for i in range(3)], dim=1).detach().numpy()
        signed = np.concatenate([p.shape.numpy(), -p.shape.numpy()], axis=1)
        conn = np.concatenate([p.mesh["interface_upper"], p.mesh["interface_lower"]], axis=1)
        surface_tangent = surface_tangent.reshape(len(p.area), len(p.surface_weights), 3, 3)
        interface = surface_matrix(conn, np.einsum("qa,fqij,qb,f,q->faibj", signed,
            surface_tangent, signed, p.area.numpy(), p.surface_weights.numpy(),
            optimize=True).reshape(len(conn), 3*conn.shape[1], 3*conn.shape[1]), p.ndof)
        current = displacement[-1].numpy()
        rods = sparse.csr_matrix(p.anchor_operator.numpy())
        engaged = np.zeros(len(p.anchor_stiffness), dtype=bool)
        if p.anchor_reference is not None:
            mean = torch.einsum("d,ad->a", displacement[-1], p.anchor_operator)
            trial = p.assembly_tension+p.anchor_stiffness*(mean-p.anchor_reference)
            # Match clamp_min's active derivative at exactly zero tension.
            # A negative trial load remains slack even though its output is zero.
            engaged = (trial >= 0).numpy()
        jack = sparse.csr_matrix(p.jack_operator.numpy()[None])
        self.local = (bulk_matrix
            +interface+p.wall_tangent(current)+p.trestle_tangent(current)
            +rods.T@sparse.diags(p.anchor_stiffness.numpy()*engaged)@rods)
        if not prescribed_force:
            self.local += jack.T@jack/problem.config["apparatus"]["compliance_m_per_n"]
        self.local = self.local.tocsc()
        super().__init__(dtype=np.dtype(float), shape=(p.ndof, p.ndof))

    def _matvec(self, vector):
        value = np.asarray(vector).reshape(-1)
        return self.local@value+self.left@(self.averaging@(self.right@value))

    def _rmatvec(self, vector):
        value = np.asarray(vector).reshape(-1)
        return self.local.T@value+self.right.T@(self.averaging.T@(self.left.T@value))

    def metric_inverse(self, vector):
        value = np.zeros(self.shape[0])
        for projection, factor in zip(self.coordinates.projections, self.coordinates.factors):
            reduced = (projection.T@np.asarray(vector))[None]
            value += projection@factor.decode(factor.whiten(reduced))[0]
        return value

    def solve(self, residual, tolerance=.02, *, direct=False):
        if direct:
            # A distant endpoint can defeat the reference-metric Krylov solve.
            # Assemble the same nonlocal blocks, without changing the equation.
            matrix = (self.local+self.left@(self.averaging@self.right)).tocsc()
            print(json.dumps(dict(event="factor_current_tangent", nonzeros=matrix.nnz)), flush=True)
            direction = splu(matrix).solve(-np.asarray(residual))
            relative = float(np.linalg.norm(self@direction+residual)/np.linalg.norm(residual))
            if not np.isfinite(relative) or relative > tolerance:
                raise RuntimeError(f"Direct current Newton equation residual: {relative}")
            return direction, dict(method="sparse_lu", relative_linear_residual=relative)
        preconditioner = LinearOperator(self.shape, matvec=self.metric_inverse, dtype=float)
        progress = []

        def record(value):
            progress.append(float(value))
            if len(progress) % 25 == 0:
                print(json.dumps(dict(newton_krylov_iterations=len(progress))), flush=True)

        direction, status = gmres(self, -np.asarray(residual), M=preconditioner,
            rtol=tolerance, atol=0., restart=120, maxiter=20,
            callback=record, callback_type="pr_norm")
        if status:
            raise RuntimeError(f"Current controlled Newton equation failed: GMRES status {status}")
        return direction, dict(iterations=len(progress),
            relative_linear_residual=float(np.linalg.norm(self@direction+residual)/np.linalg.norm(residual)))


class IndirectControlTangent(LinearOperator):
    """Border the actual controlled tangent by a local-displacement equation.

    Unknowns are spatial displacement and actuator displacement, both in m.
    The last equation is divided by the series compliance, giving N units.
    No inverse of the actual controlled tangent is required near a fold.
    """

    def __init__(self, tangent, jack, local, compliance):
        self.tangent = tangent
        self.loading = np.asarray(jack)/compliance
        self.constraint = np.asarray(local)/compliance
        self.response = tangent.metric_inverse(self.loading)
        self.schur = self.constraint@self.response
        if not np.isfinite(self.schur) or self.schur == 0:
            raise ValueError("The local coordinate does not couple to the loading metric")
        super().__init__(dtype=np.dtype(float), shape=(tangent.shape[0]+1,)*2)

    def _matvec(self, vector):
        vector = np.asarray(vector).reshape(-1)
        return np.r_[self.tangent@vector[:-1]-self.loading*vector[-1],
                     self.constraint@vector[:-1]]

    def _rmatvec(self, vector):
        vector = np.asarray(vector).reshape(-1)
        return np.r_[self.tangent.rmatvec(vector[:-1])+self.constraint*vector[-1],
                     -self.loading@vector[:-1]]

    def metric_inverse(self, vector):
        spatial = self.tangent.metric_inverse(np.asarray(vector)[:-1])
        control = (vector[-1]-self.constraint@spatial)/self.schur
        return np.r_[spatial+self.response*control, control]

    def solve(self, residual, tolerance=.02, *, preconditioner="reference"):
        if preconditioner == "current_local":
            # Track contact and degraded stiffness in the preconditioner while
            # retaining U W G in every actual Newton matrix-vector product.
            local = sparse.bmat([[self.tangent.local, sparse.csc_matrix(-self.loading[:, None])],
                                 [sparse.csc_matrix(self.constraint[None]), None]], format="csc")
            factor = splu(local)
            inverse = LinearOperator(self.shape, matvec=factor.solve, dtype=float)
        elif preconditioner == "reference":
            inverse = LinearOperator(self.shape, matvec=self.metric_inverse, dtype=float)
        else:
            raise ValueError(f"Unknown indirect preconditioner: {preconditioner}")
        progress = []

        def record(value):
            progress.append(float(value))
            if len(progress) % 25 == 0:
                print(json.dumps(dict(indirect_krylov_iterations=len(progress))), flush=True)

        direction, status = gmres(self, -np.asarray(residual), M=inverse,
            rtol=tolerance, atol=0., restart=120, maxiter=20,
            callback=record, callback_type="pr_norm")
        if status:
            raise RuntimeError(f"Indirect controlled Newton equation failed: GMRES status {status}")
        return direction, dict(iterations=len(progress), preconditioner=preconditioner,
            relative_linear_residual=float(np.linalg.norm(self@direction+residual)/np.linalg.norm(residual)))
