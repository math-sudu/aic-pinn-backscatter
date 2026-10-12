"""Local physics-corrected regional histories for shared-parameter training.

At each refresh, a complete history Newton correction defines local targets
for H=(fixed-reference host, circuit-minus-host). The shared Gauss-Newton
model stacks their derivatives with the original force Jacobian. The actual
nonlinear H-target and force objectives govern line search. This is a local
output surrogate of H' J^-1 R, not that frozen residual metric away from the
refresh point. No measured or independently solved response is a target.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.sparse.linalg import LinearOperator
import torch

from fixed_reference_history import absolute_components, parameter_indices
from history_optimization import initialize_worker, worker_direction
from shared_readout_metric import circuit_derivatives


class RegionalHistoryMetric:
    """Small observation maps retain both time-reference and material coupling."""

    def __init__(self, problem, network, parameters, readout):
        p = problem.physics
        self.problem, self.network, self.readout = problem, network, readout
        self.parameters = parameters
        self.nominal = int(parameter_indices(parameters.numpy(), [[1., -1.]])[0])
        self.changed = np.arange(len(parameters)) != self.nominal
        self.scale = problem.coordinates.coordinate_scale*p.scale
        self.maps = []
        with torch.no_grad():
            for matrix in [problem.functionals.q_matrix, problem.raw_matrix]:
                # whiten includes 1/scale; its transpose coordinate map needs scale.
                self.maps.append(problem.coordinates.whiten(matrix[None])[0]*self.scale**2)
            self.start_raw = network.raw(parameters).detach()
            self.start = self.values(self.start_raw).detach()
            self.features = (network.features(parameters)@network.feature_transform).numpy()
            absolute = self.absolute(self.start_raw)
            raw_gauges = absolute@self.maps[1].T
        self.circuit_gradient = np.stack([circuit_derivatives(readout, raw, [])[0].numpy()
                                          for raw in raw_gauges])
        self.target = None
        self.coefficients = None

    def absolute(self, raw):
        return absolute_components(raw) if self.network.path_coordinates == "reference_increments" else raw

    def values(self, raw):
        absolute = self.absolute(raw)
        host = absolute@self.maps[0].T
        circuit = self.readout(absolute@self.maps[1].T).flatten(2)
        host = host[:, 1:]-host[:, :1]
        circuit = circuit[:, 1:]-circuit[:, :1]
        return torch.stack([host, circuit-host], dim=-1)

    def terms(self, difference):
        change = difference[self.changed]-difference[self.nominal:self.nominal+1]
        return dict(host_value=difference[..., 0].square().mean(),
            host_material_change=change[..., 0].square().mean(),
            discrepancy_value=difference[..., 1].square().mean(),
            discrepancy_material_change=change[..., 1].square().mean())

    def set_correction(self, correction, physics_loss, coefficients=None):
        correction = self.problem.physics.tensor(correction)
        self.target = self.start+correction
        self.initial_terms = {key: float(value) for key, value in self.terms(correction).items()}
        self.coefficients = coefficients or {
            block: physics_loss/(self.initial_terms[f"{block}_value"]+
                                 self.initial_terms[f"{block}_material_change"])
            for block in ["host", "discrepancy"]}
        self.correction = correction.numpy()

    def objective(self, network):
        terms = self.terms(self.values(network.raw(self.parameters))-self.target)
        loss = sum(self.coefficients[block]*(terms[f"{block}_value"]+
                   terms[f"{block}_material_change"]) for block in ["host", "discrepancy"])
        return loss, {key: float(value.detach()) for key, value in terms.items()}

    def pack(self, values):
        changes = values[self.changed]-values[self.nominal]
        return np.concatenate([np.sqrt(self.coefficients[block])*part[..., axis].ravel()/
            np.sqrt(part[..., axis].size) for axis, block in enumerate(["host", "discrepancy"])
            for part in [values, changes]])

    def unpack_transpose(self, vector):
        value_shape = self.start.shape[:-1]
        change_shape = (int(self.changed.sum()), *value_shape[1:])
        values = np.zeros(self.start.shape)
        offset = 0
        for axis, block in enumerate(["host", "discrepancy"]):
            for change, shape in [(False, value_shape), (True, change_shape)]:
                size = int(np.prod(shape))
                part = vector[offset:offset+size].reshape(shape)*np.sqrt(self.coefficients[block]/size)
                offset += size
                if change:
                    values[self.changed, ..., axis] += part
                    values[self.nominal, ..., axis] -= part.sum(axis=0)
                else:
                    values[..., axis] += part
        return values

    def matvec(self, vector):
        coefficient = vector.reshape(*self.network.shape, self.features.shape[1])
        # Map feature coefficients before material interpolation, avoiding a
        # separate coordinate decode for every material history.
        outputs = []
        for matrix in self.maps:
            mapped = np.einsum("sdf,od->sfo", coefficient, matrix.numpy(), optimize=True)
            outputs.append(np.einsum("nf,sfo->nso", self.features, mapped))
        if self.network.path_coordinates == "reference_increments":
            for output in outputs:
                output[:, 1:] += output[:, :1]
        host, gauge = outputs
        circuit = np.einsum("nsog,nsg->nso", self.circuit_gradient, gauge)
        host = host[:, 1:]-host[:, :1]
        circuit = circuit[:, 1:]-circuit[:, :1]
        return self.pack(np.stack([host, circuit-host], axis=-1))

    def rmatvec(self, vector):
        values = self.unpack_transpose(vector)
        host = values[..., 0]-values[..., 1]
        circuit = values[..., 1]
        host = np.concatenate([-host.sum(axis=1, keepdims=True), host], axis=1)
        circuit = np.concatenate([-circuit.sum(axis=1, keepdims=True), circuit], axis=1)
        gauge = np.einsum("nsog,nso->nsg", self.circuit_gradient, circuit)
        if self.network.path_coordinates == "reference_increments":
            for output in [host, gauge]:
                output[:, 0] += output[:, 1:].sum(axis=1)
        coefficient = np.zeros((*self.network.shape, self.features.shape[1]))
        for cotangent, matrix in zip([host, gauge], self.maps):
            shared = np.einsum("nf,nso->sfo", self.features, cotangent)
            coefficient += np.einsum("sfo,od->sdf", shared, matrix.numpy(), optimize=True)
        return coefficient.ravel()


def build_regional_metric(problem, network, parameters, readout, physics_loss, *,
                          coefficients=None, workers=1):
    started = time.perf_counter()
    metric = RegionalHistoryMetric(problem, network, parameters, readout)
    with torch.no_grad():
        states = network(parameters).numpy()
    payloads = [(state, eta.numpy(), network.path_coordinates) for state, eta in zip(states, parameters)]
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                       initargs=(problem.setup,))
        directions = executor.map(worker_direction, payloads)
    else:
        import history_optimization
        history_optimization.worker_problem = problem
        directions = map(worker_direction, payloads)
    corrections, solves = [], []
    try:
        for i, (encoded, info) in enumerate(directions):
            # Evaluate only this case's circuit derivative, including its
            # loaded reference. No other material's state is copied here.
            direction = problem.physics.tensor(encoded)[0]
            absolute = (absolute_components(direction[None])[0]
                        if network.path_coordinates == "reference_increments" else direction)
            host = (absolute@metric.maps[0].T).numpy()
            gauges = (absolute@metric.maps[1].T).numpy()
            circuit = np.einsum("sog,sg->so", metric.circuit_gradient[i], gauges)
            host = host[1:]-host[:1]
            circuit = circuit[1:]-circuit[:1]
            corrections.append(np.stack([host, circuit-host], axis=-1))
            solves.append(info)
            print(json.dumps(dict(regional_history_target=i+1, material_conditions=len(parameters),
                seconds=time.perf_counter()-started, correction=info)), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    metric.set_correction(np.stack(corrections), physics_loss, coefficients)
    metric.work = dict(refresh_seconds=time.perf_counter()-started, coefficients=metric.coefficients,
        initial_terms=metric.initial_terms, material_directions=solves,
        local_correction_microstrain=(metric.correction*1e6).tolist(),
        initial_response_microstrain=(metric.start.numpy()*1e6).tolist(),
        target_definition="H(U_k)+H'(U_k) delta_U; J(U_k) delta_U=-R(U_k). H contains fixed-loaded-reference host and circuit-minus-host histories. Targets and block coefficients remain fixed during line search; targets refresh after each accepted update.")
    return metric


class RegionalHistoryLinearization(LinearOperator):
    def __init__(self, physics, regional):
        self.physics, self.regional = physics, regional
        self.residual = np.r_[physics.residual, regional.pack(-regional.correction)]
        self.coefficient_shape = physics.coefficient_shape
        self.residual_shape = physics.residual_shape
        self.products = dict(forward=0, transpose=0)
        super().__init__(dtype=np.dtype(float), shape=(len(self.residual), physics.shape[1]))

    def _matvec(self, vector):
        self.products["forward"] += 1
        return np.r_[self.physics@vector, self.regional.matvec(vector)]

    def _rmatvec(self, vector):
        self.products["transpose"] += 1
        n = len(self.physics.residual)
        return self.physics.rmatvec(vector[:n])+self.regional.rmatvec(vector[n:])
