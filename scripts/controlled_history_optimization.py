"""Full controlled-history Gauss--Newton updates in shared neural coefficients.

Cache the actual first derivative, including nonlocal fracture, earlier maxima,
reactive rigid balance and eliminated loads. Reverse-over-reverse products use
the cached graph; no local elastic history operator substitutes for this map.
"""

import json

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
import torch

from fixed_reference_history import absolute_components


class MaterialControlledHistory:
    def __init__(self, problem, network, parameter):
        self.problem, self.network = problem, network
        self.parameter = parameter[None]
        with torch.no_grad():
            raw = network.raw(self.parameter)[0]
        self.reference = raw[:1].detach()
        self.point = raw[1:].detach().requires_grad_(True)
        self.value = self.function(self.point)
        self.dual = torch.zeros_like(self.value, requires_grad=True)
        # This graph is linear in dual. Differentiating it with respect to dual
        # computes J v, including custom first derivatives in the physical map.
        self.pullback = torch.autograd.grad(self.value, self.point, self.dual,
                                            create_graph=True)[0]

    def field(self, point):
        raw = torch.cat([self.reference, point], dim=0)[None]
        if self.network.displacement.path_coordinates == "reference_increments":
            raw = absolute_components(raw)
        displacement = self.problem.coordinates.decode(raw)*self.problem.physics.scale
        return self.network.balance(displacement, self.parameter)

    def function(self, point):
        displacement = self.field(point)
        residual = self.problem.equilibrium(displacement, self.network.forces(displacement),
                                            self.parameter)[0]
        return self.problem.coordinates.whiten(residual)[0]

    def matvec(self, direction):
        direction = self.point.new_tensor(np.asarray(direction).reshape(self.point.shape))
        return torch.autograd.grad(self.pullback, self.dual, direction,
                                   retain_graph=True)[0].detach().numpy()

    def rmatvec(self, direction):
        direction = self.point.new_tensor(np.asarray(direction).reshape(self.value.shape))
        return torch.autograd.grad(self.value, self.point, direction,
                                   retain_graph=True)[0].detach().numpy()


class SharedControlledLinearization(LinearOperator):
    def __init__(self, problem, network, parameters):
        if network.reference_coefficients is None:
            raise ValueError("Controlled-history updates require a prepared reference")
        with torch.no_grad():
            self.features = (network.displacement.features(parameters)
                             @network.displacement.feature_transform).numpy()
        self.coefficient_shape = (len(problem.initial)-1, network.displacement.shape[-1],
                                  self.features.shape[-1])
        self.histories = []
        for eta in parameters:
            self.histories.append(MaterialControlledHistory(problem, network, eta))
            print(json.dumps(dict(controlled_history_linearization=len(self.histories),
                                  material_conditions=len(parameters))), flush=True)
        self.residual_shape = (len(parameters), *self.histories[0].value.shape)
        self.normalization = np.sqrt(np.prod(self.residual_shape))
        self.residual = np.stack([row.value.detach().numpy() for row in self.histories]
                                 ).ravel()/self.normalization
        self.products = dict(forward=0, transpose=0)
        super().__init__(dtype=np.dtype(float),
                         shape=(self.residual.size, int(np.prod(self.coefficient_shape))))

    def _matvec(self, vector):
        coefficients = np.asarray(vector).reshape(-1, self.features.shape[-1])
        values = np.stack([row.matvec(coefficients@feature)
                           for row, feature in zip(self.histories, self.features)])
        self.products["forward"] += 1
        print(json.dumps(dict(controlled_history_products=self.products)), flush=True)
        return values.ravel()/self.normalization

    def _rmatvec(self, vector):
        values = np.asarray(vector).reshape(self.residual_shape)
        result = np.zeros(self.coefficient_shape)
        for row, feature, value in zip(self.histories, self.features, values):
            result += row.rmatvec(value)[..., None]*feature
        self.products["transpose"] += 1
        return result.ravel()/self.normalization

    @torch.no_grad()
    def objective(self, coefficients):
        flat = np.asarray(coefficients).reshape(-1, self.features.shape[-1])
        return float(np.mean([row.function(row.point+row.point.new_tensor(
            (flat@feature).reshape(row.point.shape))).square().mean().item()
            for row, feature in zip(self.histories, self.features)]))

    @staticmethod
    def fold_update(coefficients):
        reference = coefficients.new_zeros((1, *coefficients.shape[1:]))
        return torch.cat([reference, coefficients], dim=0).flatten(0, 1)


def controlled_history_direction(problem, network, parameters, tolerance):
    linear = SharedControlledLinearization(problem, network, parameters)
    result = lsmr(linear, -linear.residual, atol=tolerance, btol=tolerance)
    update, stop, iterations, residual_norm, normal_norm, operator_norm, condition, norm = result
    image = linear@update
    details = dict(method="lsmr", tolerance=tolerance, stop_code=int(stop), iterations=int(iterations),
        products=linear.products.copy(), coefficient_shape=linear.coefficient_shape,
        residual_shape=linear.residual_shape,
        feature_singular_values=np.linalg.svd(linear.features, compute_uv=False).tolist(),
        objective_before=float(linear.residual@linear.residual),
        linearized_objective_after=float(np.sum((linear.residual+image)**2)),
        directional_derivative=float(2*linear.residual@image), residual_norm=float(residual_norm),
        normal_residual_norm=float(normal_norm), operator_norm_estimate=float(operator_norm),
        condition_estimate=float(condition), update_norm=float(norm))
    return update.reshape(linear.coefficient_shape), linear, details
