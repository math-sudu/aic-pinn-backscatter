"""Full-force history updates in one shared neural output layer."""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr, minres
import torch

from fixed_reference_history import MaterialHistoryProblem, fixed_reference_components


class SharedHistoryLinearization(LinearOperator):
    """Jacobian of the original normalized force residual in shared coefficients.

    The intact coordinate and residual scales cancel in the Jacobian, but not
    in the residual itself. Decode only the feature coefficients; accumulate
    transposed physical forces across materials before their coordinate map.
    """

    def __init__(self, problem, network, parameters):
        self.coordinates = problem.coordinates
        self.path_coordinates = network.path_coordinates
        self.path_shape = network.shape
        with torch.no_grad():
            self.features = (network.features(parameters)@network.feature_transform).numpy()
        self.coefficient_shape = (*self.path_shape, self.features.shape[1])
        self.residual_shape = (len(parameters), *self.path_shape)
        self.normalization = np.sqrt(np.prod(self.residual_shape))
        self.boundaries = np.cumsum(self.coordinates.sizes)[:-1]
        self.linearizations, residuals = [], []
        for eta in parameters:
            with torch.no_grad():
                state = network(eta[None])[0]
                force = problem.equilibrium(state[None], eta[None])[0]
                residuals.append(self.coordinates.whiten(force)[0].numpy())
            # Full physical matrices avoid duplicating stiffness entries across
            # the four parity sectors. Both coordinate maps retain all sectors.
            self.linearizations.append(problem.linearization(state, eta))
            print(json.dumps(dict(history_linearization=len(self.linearizations),
                                  material_conditions=len(parameters))), flush=True)
        self.residual = np.stack(residuals).ravel()/self.normalization
        self.products = dict(forward=0, transpose=0)
        super().__init__(dtype=np.dtype(float),
                         shape=(self.residual.size, int(np.prod(self.coefficient_shape))))

    def decode(self, values):
        flat = values.reshape(-1, self.path_shape[-1])
        reduced = np.concatenate([factor.decode(part) for factor, part in
            zip(self.coordinates.factors, np.split(flat, self.boundaries, axis=-1))], axis=-1)
        return (self.coordinates.projection@reduced.T).T.reshape(values.shape)

    def whiten(self, values):
        flat = values.reshape(-1, self.path_shape[-1])
        reduced = (self.coordinates.projection.T@flat.T).T
        return np.concatenate([factor.whiten(part) for factor, part in
            zip(self.coordinates.factors, np.split(reduced, self.boundaries, axis=-1))], axis=-1
            ).reshape(values.shape)

    def _matvec(self, value):
        coefficients = value.reshape(self.coefficient_shape).transpose(2, 0, 1).copy()
        if self.path_coordinates == "reference_increments":
            coefficients[:, 1:] += coefficients[:, :1]
        physical = self.decode(coefficients)
        states = (self.features@physical.reshape(len(physical), -1)).reshape(self.residual_shape)
        forces = np.stack([linear.apply(state) for linear, state in zip(self.linearizations, states)])
        self.products["forward"] += 1
        if self.products["forward"] % 5 == 0:
            print(json.dumps(dict(shared_history_products=self.products)), flush=True)
        return self.whiten(forces).ravel()/self.normalization

    def _rmatvec(self, value):
        forces = self.decode(value.reshape(self.residual_shape))
        states = np.stack([linear.apply_transpose(force)
                           for linear, force in zip(self.linearizations, forces)])
        physical = (self.features.T@states.reshape(len(states), -1)).reshape(
            self.features.shape[1], *self.path_shape)
        coefficients = self.whiten(physical)
        if self.path_coordinates == "reference_increments":
            coefficients[:, 0] += coefficients[:, 1:].sum(axis=1)
        self.products["transpose"] += 1
        return coefficients.transpose(1, 2, 0).ravel()/self.normalization


def gauss_newton_history_direction(problem, network, parameters, tolerance, regional=None):
    """Inexact least squares in shared parameters, with the full history metric."""
    linear = SharedHistoryLinearization(problem, network, parameters)
    if regional is not None:
        from regional_history_training import RegionalHistoryLinearization
        linear = RegionalHistoryLinearization(linear, regional)
    result = lsmr(linear, -linear.residual, atol=tolerance, btol=tolerance)
    update, stop, iterations, residual_norm, normal_norm, operator_norm, condition, norm = result
    image = linear@update
    details = dict(method="lsmr", tolerance=tolerance, stop_code=int(stop), iterations=int(iterations),
        products=linear.products, residual_shape=linear.residual_shape,
        coefficient_shape=linear.coefficient_shape,
        objective_before=float(linear.residual@linear.residual),
        linearized_objective_after=float(np.sum((linear.residual+image)**2)),
        directional_derivative=float(2*linear.residual@image),
        residual_norm=float(residual_norm), normal_residual_norm=float(normal_norm),
        operator_norm_estimate=float(operator_norm), condition_estimate=float(condition),
        update_norm=float(norm))
    print(json.dumps(dict(shared_history_least_squares=details)), flush=True)
    return problem.physics.tensor(update.reshape(-1, linear.coefficient_shape[-1])), details


def initialize_worker(setup):
    global worker_problem
    torch.set_num_threads(1)
    worker_problem = MaterialHistoryProblem(setup)


def worker_direction(payload):
    state, eta, path_coordinates = payload
    p = worker_problem.physics
    physical, _, info = history_correction(worker_problem, p.tensor(state), p.tensor(eta))
    encoded = p.tensor(worker_problem.coordinates.encode_nominal(physical))[None]
    if path_coordinates == "reference_increments":
        encoded = fixed_reference_components(encoded)
    return encoded.numpy(), info


def history_correction(problem, state, eta):
    """One frozen Newton direction at a neural path; retain all temporal blocks."""
    p, coordinates = problem.physics, problem.coordinates
    projection = coordinates.projection
    with torch.no_grad():
        force = problem.equilibrium(state[None], eta[None])[0][0].numpy()
    linear = problem.linearization(state, eta, projection=projection)
    right = -(projection.T@force.T).T
    boundaries = np.cumsum(coordinates.sizes)[:-1]

    def precondition(value):
        return np.concatenate([factor.decode(factor.whiten(part))
            for factor, part in zip(coordinates.factors, np.split(value, boundaries))])

    metric = LinearOperator((linear.nfree, linear.nfree), matvec=precondition, dtype=float)
    correction, products = [], []
    for n in range(linear.nstep):
        rhs = right[n].copy()
        for m in range(n):
            rhs -= linear.blocks[n][m]@correction[m]
        count = [0]

        def callback(_):
            count[0] += 1

        value, status = minres(linear.blocks[n][n], rhs, M=metric, rtol=1e-9, callback=callback)
        if status:
            raise RuntimeError(f"History-direction linear solve failed at step {n}: {status}")
        correction.append(value)
        products.append(count[0])
    correction = np.stack(correction)
    tangent = linear.apply(correction)
    details = dict(iterations_by_step=products,
        linear_equation_relative_residual=float(np.linalg.norm(tangent-right)/np.linalg.norm(right)))
    return (projection@correction.T).T, (projection@tangent.T).T, details


def projected_history_direction(problem, network, parameters, workers=1):
    """Least-squares projection of physical corrections into shared features."""
    with torch.no_grad():
        features = (network.features(parameters)@network.feature_transform).numpy()
    inverse_features = np.linalg.pinv(features)
    last = network.layers[-1]
    update = torch.zeros((last.out_features, last.in_features+1), dtype=last.weight.dtype,
                         device=last.weight.device)
    details = []

    def serial_directions():
        for eta in parameters:
            with torch.no_grad():
                state = network(eta[None])[0]
            physical, _, info = history_correction(problem, state, eta)
            encoded = problem.physics.tensor(problem.coordinates.encode_nominal(physical))[None]
            if network.path_coordinates == "reference_increments":
                encoded = fixed_reference_components(encoded)
            yield encoded, info

    executor = None
    if workers == 1:
        directions = serial_directions()
    else:
        with torch.no_grad():
            states = network(parameters).numpy()
        executor = ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                                       initargs=(problem.setup,))
        directions = executor.map(worker_direction,
            ((state, eta.numpy(), network.path_coordinates) for state, eta in zip(states, parameters)))
    try:
        for i, (encoded, info) in enumerate(directions):
            update += problem.physics.tensor(encoded).reshape(-1, 1)*problem.physics.tensor(inverse_features[:, i])[None]
            details.append(dict(parameters=parameters[i].tolist(), **info))
            print(json.dumps(dict(history_direction=i+1, material_conditions=len(parameters),
                                  linear_iterations=sum(info["iterations_by_step"]))), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    return update, details
