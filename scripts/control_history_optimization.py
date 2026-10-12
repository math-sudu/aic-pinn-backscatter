"""Full causal Gauss--Newton products for the shared control-history head.

The feature SVD removes only coefficient null directions on the training
protocol. Its inverse maps the update back into the existing common head;
queries continue to use their own controls and recurrent features.
"""

import json

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
import torch


class ControlHistoryLinearization(LinearOperator):
    def __init__(self, network, parameters, prefix, prefix_forces, controls, regional=None):
        self.network, self.parameters = network, parameters
        self.prefix, self.prefix_forces, self.controls = prefix, prefix_forces, controls
        self.regional = regional
        with torch.no_grad():
            features = network.features(parameters, prefix, controls)
            augmented = torch.cat([features, torch.ones_like(features[..., :1])], dim=-1)
            self.features = augmented.reshape(-1, augmented.shape[-1]).numpy()
            raw = network.head(features)
        u, singular, vh = np.linalg.svd(self.features, full_matrices=False)
        rank = int(np.sum(singular > np.finfo(float).eps*max(self.features.shape)*singular[0]))
        self.active = u[:, :rank]
        self.inverse_features = vh[:rank].T/singular[:rank]
        self.singular = singular
        self.raw_shape = raw.shape
        self.coefficient_shape = (rank, raw.shape[-1])
        self.point = raw.detach().requires_grad_(True)
        self.value = self.function(self.point)
        self.dual = torch.zeros_like(self.value, requires_grad=True)
        self.pullback = torch.autograd.grad(self.value, self.point, self.dual, create_graph=True)[0]
        self.normalization = np.sqrt(self.value.numel()) if regional is None else 1.
        self.residual = self.value.detach().numpy().ravel()/self.normalization
        self.products = dict(forward=0, transpose=0)
        super().__init__(dtype=np.dtype(float), shape=(self.value.numel(), int(np.prod(self.coefficient_shape))))

    def function(self, point):
        n, p = self.network, self.network.problem
        fields, forces = n.field_from_raw(point, self.parameters, self.prefix, self.prefix_forces, self.controls)
        residual = p.equilibrium(fields, forces, self.parameters)[0][:, self.prefix.shape[1]:]
        whitened = p.coordinates.whiten(residual)
        if self.regional is None:
            return whitened
        return torch.cat([whitened.flatten()/np.sqrt(whitened.numel()),
            self.regional.pack(self.regional.values(fields)-self.regional.target)])

    def raw_direction(self, vector):
        return (self.active@np.asarray(vector).reshape(self.coefficient_shape)).reshape(self.raw_shape)

    def _matvec(self, vector):
        direction = self.point.new_tensor(self.raw_direction(vector))
        image = torch.autograd.grad(self.pullback, self.dual, direction, retain_graph=True)[0]
        self.products["forward"] += 1
        if self.products["forward"] % 10 == 0:
            print(json.dumps(dict(control_history_products=self.products)), flush=True)
        return image.detach().numpy().ravel()/self.normalization

    def _rmatvec(self, vector):
        cotangent = self.value.new_tensor(np.asarray(vector).reshape(self.value.shape))
        pullback = torch.autograd.grad(self.value, self.point, cotangent, retain_graph=True)[0]
        result = self.active.T@pullback.detach().numpy().reshape(-1, self.point.shape[-1])
        self.products["transpose"] += 1
        return result.ravel()/self.normalization

    def head_direction(self, vector):
        coefficients = np.asarray(vector).reshape(self.coefficient_shape)
        return self.inverse_features@coefficients


def control_history_direction(network, parameters, prefix, prefix_forces, controls, *,
                              tolerance=.02, maxiter=60, regional=None):
    linear = ControlHistoryLinearization(network, parameters, prefix, prefix_forces, controls, regional)
    solution = lsmr(linear, -linear.residual, atol=tolerance, btol=tolerance, maxiter=maxiter)
    update, stop, iterations = solution[:3]
    image = linear@update
    details = dict(method="shared_head_gauss_newton", solver="lsmr", stop_code=int(stop),
        iterations=int(iterations), tolerance=tolerance, products=linear.products, feature_rank=linear.coefficient_shape[0],
        feature_singular_values=linear.singular.tolist(),
        objective_before=float(linear.residual@linear.residual),
        linearized_objective_after=float(np.sum((linear.residual+image)**2)),
        directional_derivative=float(2*linear.residual@image))
    return linear.head_direction(update), details
