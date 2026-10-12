"""Sparse, differentiable intact-elastic coordinates for finite spatial paths.

The factor is a coordinate/force metric. Nonlinear states still come from the
neural network trained with the complete cohesive/contact history residual.
"""

import copy
import itertools

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, splu
import torch

from spatial_paths import reflection_maps
from spatial_resolution import reflection_projection


class SparseCholesky:
    """A symmetric unpivoted sparse LU yields P K P.T = L D L.T."""

    def __init__(self, matrix):
        factor = splu(matrix.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0,
                      options={"SymmetricMode": True})
        if not np.array_equal(factor.perm_r, factor.perm_c) or np.any(factor.U.diagonal() <= 0):
            raise ValueError("The intact metric did not admit symmetric positive factorization")
        self.permutation = factor.perm_r
        self.inverse = np.argsort(self.permutation)
        diagonal = factor.U.diagonal()
        error = factor.U-sparse.diags(diagonal)@factor.L.T
        if np.linalg.norm(error.data) > 1e-9*np.linalg.norm(factor.U.data):
            raise ValueError("Sparse factors do not form the required LDL transpose pair")
        self.root = np.sqrt(diagonal)
        lower = factor.L.tocsc()
        self.upper = lower.T.tocsr()
        # Reuse triangular solve setup during every neural forward/backward
        # call, instead of rebuilding it for each history vector.
        self.triangular = splu(lower, permc_spec="NATURAL", diag_pivot_thresh=0)

    def decode(self, values):
        flat = values.reshape(-1, len(self.root))
        solved = self.triangular.solve((flat/self.root).T, trans="T")
        return solved.T[:, self.permutation].reshape(values.shape)

    def whiten(self, values):
        flat = values.reshape(-1, len(self.root))
        solved = self.triangular.solve(flat[:, self.inverse].T)
        return (solved.T/self.root).reshape(values.shape)

    def encode(self, values):
        flat = values.reshape(-1, len(self.root))
        return ((self.upper@flat[:, self.inverse].T).T*self.root).reshape(values.shape)


class FactorTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, factor, decode):
        ctx.factor, ctx.decode = factor, decode
        result = (factor.decode if decode else factor.whiten)(values.detach().cpu().numpy())
        return torch.as_tensor(result, dtype=values.dtype, device=values.device)

    @staticmethod
    def backward(ctx, gradient):
        # The two maps are exact transposes, including the permutation.
        return FactorTransform.apply(gradient, ctx.factor, not ctx.decode), None, None


class SpatialCoordinates:
    """Retain every sector allowed by the prescribed field's actual symmetry.

    The fixed elastic metric decouples in parity sectors even when the nonlinear
    path does not. Forces are always evaluated in the full domain.
    """

    def __init__(self, physics, bond_scales=None, *, parities=None, metric=None):
        p = self.physics = physics
        self.reflections = [reflection_maps(p, axis)[2] for axis in (0, 1)]
        self.parities = self.field_parities(bond_scales) if parities is None else list(parities)
        free = p.free.cpu().numpy()
        matrix = (p.elastic_stiffness() if metric is None else metric)[free][:, free]
        self.projections, self.lifts, self.factors, self.sizes = [], [], [], []
        for parity in self.parities:
            projection = reflection_projection(p, parity)
            factor = SparseCholesky(projection.T@matrix@projection)
            coo = projection.tocoo()
            lift = torch.sparse_coo_tensor(p.tensor(np.stack([coo.row, coo.col]), torch.long),
                p.tensor(coo.data), coo.shape, device=p.device).coalesce()
            self.projections.append(projection)
            self.lifts.append(lift)
            self.factors.append(factor)
            self.sizes.append(projection.shape[1])
        self.projection = sparse.hstack(self.projections, format="csr")
        obs = p.settings["observations"]["regions_m"]
        self.coordinate_scale = np.sqrt(p.settings["bulk"]["top_young_pa"]*
                                        (max(r[1] for r in obs)-min(r[0] for r in obs)))

    def field_parities(self, bond_scales):
        options = []
        for quadrature in self.reflections:
            symmetric = np.allclose(bond_scales[..., quadrature], bond_scales, rtol=0, atol=1e-13)
            options.append((1,) if symmetric else (1, -1))
        return list(itertools.product(*options))

    def for_field(self, bond_scale):
        """Reuse factors while retaining only symmetries this endpoint has."""
        selected = copy.copy(self)
        selected.parities = self.field_parities(bond_scale)
        indices = [self.parities.index(parity) for parity in selected.parities]
        for name in ("projections", "lifts", "factors", "sizes"):
            setattr(selected, name, [getattr(self, name)[i] for i in indices])
        selected.projection = sparse.hstack(selected.projections, format="csr")
        return selected

    def decode(self, raw):
        result = raw.new_zeros((*raw.shape[:-1], len(self.physics.free)))
        for values, lift, factor in zip(torch.split(raw, self.sizes, dim=-1), self.lifts, self.factors):
            decoded = FactorTransform.apply(values, factor, True)
            result += torch.sparse.mm(lift, decoded.flatten(0, 1).T).T.reshape(result.shape)
        return result*self.coordinate_scale

    def whiten(self, force):
        parts = []
        for lift, factor, size in zip(self.lifts, self.factors, self.sizes):
            reduced = torch.sparse.mm(lift.T, force.flatten(0, 1).T).T.reshape(*force.shape[:-1], size)
            parts.append(FactorTransform.apply(reduced, factor, False))
        return torch.cat(parts, dim=-1)/(self.coordinate_scale*self.physics.scale)

    def encode_nominal(self, displacement):
        free = displacement[:, self.physics.free.cpu().numpy()]/(self.physics.scale*self.coordinate_scale)
        return np.concatenate([factor.encode((projection.T@free.T).T)
            for projection, factor in zip(self.projections, self.factors)], axis=-1)

    def correct_path(self, linear, rhs):
        """One residual RHS per step, preconditioned by the intact metric.

        This avoids factoring the large coupled finite-field Jacobian just to
        obtain one diagnostic correction. Earlier-time blocks remain explicit.
        """
        boundaries = np.cumsum(self.sizes)[:-1]

        def precondition(value):
            return np.concatenate([factor.decode(factor.whiten(part))
                for factor, part in zip(self.factors, np.split(value, boundaries))])

        size = sum(self.sizes)
        metric = LinearOperator((size, size), matvec=precondition, dtype=float)
        result = []
        for n, value in enumerate(rhs):
            value = value.copy()
            for m in range(n):
                value -= linear.blocks[n][m]@result[m]
            correction, status = gmres(linear.blocks[n][n], value, M=metric,
                                        rtol=1e-8, atol=0, restart=80)
            if status:
                raise RuntimeError(f"Path correction solve failed at step {n}: {status}")
            result.append(correction)
        return np.stack(result)
