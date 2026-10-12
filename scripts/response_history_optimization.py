"""Shared physical-residual updates in selected deformation directions.

Observable gradients or current path shapes select a subspace of the existing
output coefficients. Measured responses never enter its objective or line search.
"""

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from controlled_history_optimization import MaterialControlledHistory
from fixed_reference_history import absolute_components, ROOT


class BranchResponseBasis:
    """Coordinate Riesz directions for the two equal-load acquisition stages."""

    def __init__(self, problem, network):
        self.problem, self.network = problem, network
        labels = list(problem.boundary["stage_labels"])
        self.rising = labels.index("rising_300kn")
        self.falling = labels.index("falling_300kn")
        settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
        q = problem.base.functionals.q_matrix.reshape(2, 3, 2, -1)
        operators = [problem.physics.relative_operator.mean(0)]
        self.names = ["local_mean_branch_difference_mm"]
        for direction in range(2):
            for region in (0, 2):
                interval = settings["regions"][region]["interval_m"]
                operators.append(q[direction, region, 0]*(interval[1]-interval[0]))
                self.names.append(f"direction_{direction+1}_{settings['regions'][region]['name']}_extension_branch_difference_mm")
        self.operators = torch.stack(operators)*1e3
        parameter = problem.physics.tensor([1., -1.])
        row = MaterialControlledHistory(problem, network, parameter)
        values = self.responses(row.field(row.point))
        gradients = np.stack([torch.autograd.grad(value, row.point, retain_graph=True)[0]
                              .detach().numpy().ravel() for value in values])
        self.point_shape = tuple(row.point.shape)
        # Scaling each nonzero functional affects neither its span nor the update.
        norms = np.linalg.norm(gradients, axis=1)
        if np.any(norms == 0):
            raise ValueError("A selected branch response has no deformation derivative")
        left, singular, _ = np.linalg.svd((gradients/norms[:, None]).T, full_matrices=False)
        rank = np.count_nonzero(singular > singular[0]*max(gradients.shape)*np.finfo(float).eps)
        self.vectors = left[:, :rank]
        self.details = dict(parameter=parameter.tolist(), names=self.names,
            stages=[labels[self.rising], labels[self.falling]],
            values_mm=values.detach().tolist(), gradient_norms_mm=norms.tolist(),
            normalized_gradient_singular_values=singular.tolist(), rank=int(rank),
            response_jacobian_mm=(gradients@self.vectors).tolist(),
            role="Riesz span in the existing neural coordinates; observables choose directions, not response labels")
        print(json.dumps(dict(response_basis=self.details)), flush=True)

    def responses(self, displacement):
        return (displacement[0, self.falling]-displacement[0, self.rising])@self.operators.T

    def expand(self, coefficients):
        return (self.vectors@np.asarray(coefficients)).reshape(*self.point_shape, -1)


class PathAmplitudeBasis:
    """Independent amplitudes of the current deformation at every later state.

    Remove the constituent rigid-motion nullspace before normalizing. The
    implicit equilibrium layer supplies those motions again after deformation
    changes. Time-local columns retain their full causal residual images.
    """

    def __init__(self, problem, network):
        if network.constituent_balance is None:
            raise ValueError("Path amplitudes require constituent equilibrium")
        self.problem, self.network = problem, network
        parameter = problem.physics.tensor([1., -1.])
        with torch.no_grad():
            raw = network.raw(parameter[None])[0].numpy()
        increments = raw[1:].copy()
        if network.displacement.path_coordinates != "reference_increments":
            increments -= raw[:1]
        encoded = problem.coordinates.encode_nominal(network.constituent_balance.modes.T.numpy()).T
        motion, _ = np.linalg.qr(encoded, mode="reduced")
        deformation = increments-(increments@motion)@motion.T
        self.norms = np.linalg.norm(deformation, axis=1)
        if np.any(self.norms == 0):
            raise ValueError("A path state has no deformation amplitude direction")
        self.point_shape = deformation.shape
        self.spatial = deformation/self.norms[:, None]
        self.vectors = np.zeros((deformation.size, len(deformation)))
        blocks = self.vectors.reshape(*self.point_shape, -1)
        for stage, vector in enumerate(self.spatial):
            blocks[stage, :, stage] = vector
        self.details = dict(parameter=parameter.tolist(),
            stages=list(map(str, problem.boundary["stage_labels"][1:])),
            deformation_coordinate_norms=self.norms.tolist(),
            removed_rigid_coordinate_norms=np.linalg.norm(increments-deformation, axis=1).tolist(),
            rank=len(deformation),
            role="Time-local amplitudes of the current nominal deformation, shared across material features; no response labels")
        print(json.dumps(dict(path_amplitude_basis=self.details)), flush=True)

    def expand(self, coefficients):
        return self.spatial[:, :, None]*np.asarray(coefficients)[:, None, :]


def reduce_material(images, residual, feature):
    """Loss-preserving QR reduction of one material's tall subspace Jacobian."""
    orthogonal, triangular = np.linalg.qr(images, mode="reduced")
    projected = orthogonal.T@residual
    shared = (triangular[:, :, None]*feature[None, None, :]).reshape(len(triangular), -1)
    return shared, projected


def material_response_reduction(problem, network, parameter, feature, vectors):
    row = MaterialControlledHistory(problem, network, problem.physics.tensor(parameter))
    images = np.column_stack([row.matvec(vector).ravel() for vector in vectors.T])
    residual = row.value.detach().numpy().ravel()
    matrix, value = reduce_material(images, residual, feature)
    return matrix, value, float(residual@residual)


def initialize_response_worker(configuration, state, fixed_reference, vectors):
    from controlled_fracture_history import ControlledFractureProblem
    global response_worker_problem, response_worker_network, response_worker_vectors
    torch.set_num_threads(2)
    response_worker_problem = ControlledFractureProblem(configuration)
    response_worker_network = response_worker_problem.load_network(
        exact_control=True, fixed_reference=fixed_reference)
    response_worker_network.load_state_dict(state)
    response_worker_vectors = vectors


def response_worker_reduction(payload):
    parameter, feature = payload
    return material_response_reduction(response_worker_problem, response_worker_network,
                                       parameter, feature, response_worker_vectors)


class SharedResponseLinearization:
    def __init__(self, problem, network, parameters, basis, workers=1):
        self.problem, self.network, self.parameters, self.basis = problem, network, parameters, basis
        with torch.no_grad():
            self.raw = network.raw(parameters).detach()
            self.features = (network.displacement.features(parameters)
                             @network.displacement.feature_transform).numpy()
        self.coefficient_shape = (*basis.point_shape, self.features.shape[-1])
        self.reduced_shape = (basis.vectors.shape[1], self.features.shape[-1])
        self.normalization_squared = len(parameters)*len(problem.initial)*problem.physics.ndof
        matrices, projected, squared = [], [], []
        payloads = zip(parameters.numpy(), self.features)
        pool = None
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers, initializer=initialize_response_worker,
                initargs=(problem.config, network.state_dict(), network.reference_coefficients is not None,
                          basis.vectors))
            results = pool.map(response_worker_reduction, payloads)
        else:
            results = (material_response_reduction(problem, network, eta, feature, basis.vectors)
                       for eta, feature in payloads)
        try:
            for index, (matrix, value, norm_squared) in enumerate(results):
                matrices.append(matrix)
                projected.append(value)
                squared.append(norm_squared)
                print(json.dumps(dict(response_history_linearization=index+1,
                    material_conditions=len(parameters), directions=basis.vectors.shape[1])), flush=True)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        scale = np.sqrt(self.normalization_squared)
        self.matrix = np.vstack(matrices)/scale
        self.projected = np.concatenate(projected)/scale
        self.objective_before = float(sum(squared)/self.normalization_squared)

    @torch.no_grad()
    def objective(self, coefficients):
        flat = np.asarray(coefficients).reshape(-1, self.features.shape[-1])
        values = []
        for raw, feature, eta in zip(self.raw, self.features, self.parameters):
            update = raw.new_tensor((flat@feature).reshape(self.basis.point_shape))
            trial = torch.cat([raw[:1], raw[1:]+update], dim=0)[None]
            if self.network.displacement.path_coordinates == "reference_increments":
                trial = absolute_components(trial)
            field = self.problem.coordinates.decode(trial)*self.problem.physics.scale
            field = self.network.balance(field, eta[None])
            residual = self.problem.equilibrium(field, self.network.forces(field), eta[None])[0]
            values.append(float(self.problem.coordinates.whiten(residual).square().mean()))
        return float(np.mean(values))


def response_history_direction(problem, network, parameters, *, path_amplitudes=False, workers=1):
    if network.reference_coefficients is None:
        raise ValueError("Response history updates require a prepared reference")
    basis = (PathAmplitudeBasis if path_amplitudes else BranchResponseBasis)(problem, network)
    linear = SharedResponseLinearization(problem, network, parameters, basis, workers)
    coefficients, _, rank, singular = np.linalg.lstsq(linear.matrix, -linear.projected, rcond=None)
    image = linear.matrix@coefficients
    increment = float(2*linear.projected@image+image@image)
    nominal_feature = linear.features[np.flatnonzero(np.all(parameters.numpy() == [1., -1.], axis=1))[0]]
    reduced = coefficients.reshape(linear.reduced_shape)
    details = dict(method="path_amplitude_qr" if path_amplitudes else "response_subspace_qr",
        response_basis=basis.details,
        coefficient_shape=linear.coefficient_shape, reduced_coefficient_shape=linear.reduced_shape,
        reduced_rank=int(rank), reduced_singular_values=singular.tolist(),
        objective_before=linear.objective_before,
        linearized_objective_after=linear.objective_before+increment,
        directional_derivative=float(2*linear.projected@image),
        update_norm=float(np.linalg.norm(coefficients)))
    if path_amplitudes:
        details["nominal_deformation_amplitude_increment"] = (reduced@nominal_feature/basis.norms).tolist()
    else:
        details["nominal_linearized_response_change_mm"] = (
            np.asarray(basis.details["response_jacobian_mm"])@reduced@nominal_feature).tolist()
    return basis.expand(reduced), linear, details
