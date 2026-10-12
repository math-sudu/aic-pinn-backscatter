"""Geometric deformation directions for extension across the cone perimeter.

These are trial fields in the existing P2 displacement space, not displacement
labels or prescribed cracks. Their amplitudes must be determined by the coupled
physical history before they can be used as structural predictions.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from controlled_history_optimization import SharedControlledLinearization


def opening_fields(mesh, radius, half_width, depths, *, depth_kinematics="radial"):
    """Eight independent side/mean/depth directions, per metre of amplitude.

    The smooth radial step connects inner and outer in-plane displacements.
    Four angular partitions select the two sides of the two measurement axes;
    their sum is one. The depth factors are mean and signed half-difference at
    the two observation depths. Flexure couples the depth field to the vertical
    displacement from Psi = A G / h, with G' = S. Its continuous transverse
    shear cancels; nodal P2 interpolation defines the actual discrete field.
    Only the top FRC body is changed.
    """
    if not 0 < half_width < radius or not depths[0] < depths[1]:
        raise ValueError("Opening directions require positive radius, width and depth separation")
    if depth_kinematics not in ("radial", "flexure"):
        raise ValueError(f"Unknown depth kinematics: {depth_kinematics}")
    xyz = mesh["xyz"]
    top = np.unique(mesh["tetra"][mesh["body"] == 0])
    r = np.linalg.norm(xyz[:, :2], axis=-1)
    normal = np.divide(xyz[:, :2], r[:, None], out=np.zeros((len(xyz), 2)), where=r[:, None] > 0)
    t = np.clip((r-radius+half_width)/(2*half_width), 0., 1.)
    step = t**3*(10+t*(-15+6*t))
    integral = 2*half_width*t**4*(2.5+t*(-3+t))+np.maximum(r-radius-half_width, 0.)
    half_depth = (depths[1]-depths[0])/2
    depth_factor = (xyz[:, 2]-np.mean(depths))/half_depth
    fields, names = [], []
    for direction in (0, 1):
        for side, sign in (("left", -1), ("right", 1)):
            angular = normal[:, direction]**2*(1+sign*normal[:, direction])/2
            tangent = np.eye(2)[direction]-normal[:, direction, None]*normal
            angular_gradient = np.divide(
                (normal[:, direction]+1.5*sign*normal[:, direction]**2)[:, None]*tangent,
                r[:, None], out=np.zeros_like(normal), where=r[:, None] > 0)
            for component, factor in (("mean", np.ones(len(xyz))), ("depth_difference", depth_factor)):
                field = np.zeros_like(xyz)
                field[top, :2] = (step*angular*factor)[top, None]*normal[top]
                if component == "depth_difference" and depth_kinematics == "flexure":
                    field[top, :2] += (integral*factor)[top, None]*angular_gradient[top]
                    field[top, 2] = -(angular*integral)[top]/half_depth
                fields.append(field.ravel())
                names.append(f"direction_{direction+1}_{side}_{component}")
    return np.asarray(fields), names


class RegionalOpeningHistoryBasis:
    """Perimeter extension at each later state, in shared neural coordinates."""

    def __init__(self, problem, network, *, depth_kinematics="radial"):
        if network.constituent_balance is None:
            raise ValueError("Regional openings require constituent equilibrium")
        p = problem.physics
        fields, names = opening_fields(p.mesh,
            p.settings["geometry"]["cone_top_radius_m"], problem.config["nonlocal_radius_m"],
            p.settings["observations"]["depths_m"], depth_kinematics=depth_kinematics)
        encoded = problem.coordinates.encode_nominal(fields).T
        rigid = problem.coordinates.encode_nominal(network.constituent_balance.modes.T.numpy()).T
        motion, _ = np.linalg.qr(rigid, mode="reduced")
        deformation = encoded-motion@(motion.T@encoded)
        spatial, self.mode_coordinates = np.linalg.qr(deformation, mode="reduced")
        self.spatial = spatial.T
        self.point_shape = (len(problem.initial)-1, len(spatial))
        self.rank = self.point_shape[0]*len(fields)
        self.details = dict(names=names, depth_kinematics=depth_kinematics,
            stages=list(map(str, problem.boundary["stage_labels"][1:])),
            spatial_rank=int(np.linalg.matrix_rank(self.mode_coordinates)),
            rank=self.rank,
            mode_coordinates_per_m=self.mode_coordinates.tolist(),
            radius_m=p.settings["geometry"]["cone_top_radius_m"],
            half_width_m=problem.config["nonlocal_radius_m"],
            role="Eight geometric extension/depth modes at every later state, shared over material features; physical residual only")

    def expand(self, coefficients):
        reduced = np.asarray(coefficients).reshape(self.point_shape[0], len(self.spatial), -1)
        return np.einsum("kd,skf->sdf", self.spatial, reduced)

    def physical_amplitudes(self, coefficients):
        """Recover signed amplitudes of the original eight fields, in metres."""
        reduced = np.asarray(coefficients).reshape(self.point_shape[0], len(self.spatial))
        return np.linalg.solve(self.mode_coordinates, reduced.T).T


class SharedRegionalOpeningLinearization(LinearOperator):
    """Restrict the actual complete-history Jacobian to the opening subspace."""

    def __init__(self, problem, network, parameters, basis):
        self.full = SharedControlledLinearization(problem, network, parameters)
        self.basis = basis
        self.reduced_shape = (basis.rank, self.full.features.shape[-1])
        super().__init__(dtype=np.dtype(float),
                         shape=(self.full.shape[0], int(np.prod(self.reduced_shape))))

    def _matvec(self, vector):
        coefficients = self.basis.expand(np.asarray(vector).reshape(self.reduced_shape))
        return self.full@coefficients.ravel()

    def _rmatvec(self, vector):
        coefficients = self.full.rmatvec(vector).reshape(self.full.coefficient_shape)
        return np.einsum("kd,sdf->skf", self.basis.spatial, coefficients).ravel()


def opening_history_direction(problem, network, parameters, tolerance, *, depth_kinematics="radial"):
    basis = RegionalOpeningHistoryBasis(problem, network, depth_kinematics=depth_kinematics)
    linear = SharedRegionalOpeningLinearization(problem, network, parameters, basis)
    full = linear.full
    result = lsmr(linear, -full.residual, atol=tolerance, btol=tolerance)
    update, stop, iterations, residual_norm, normal_norm, operator_norm, condition, norm = result
    image = linear@update
    reduced = update.reshape(linear.reduced_shape)
    nominal = np.flatnonzero(np.all(parameters.numpy() == [1., -1.], axis=1))[0]
    details = dict(method="regional_opening_lsmr", tolerance=tolerance,
        response_basis=basis.details, coefficient_shape=full.coefficient_shape,
        reduced_coefficient_shape=linear.reduced_shape,
        stop_code=int(stop), iterations=int(iterations), products=full.products.copy(),
        objective_before=float(full.residual@full.residual),
        linearized_objective_after=float(np.sum((full.residual+image)**2)),
        directional_derivative=float(2*full.residual@image),
        residual_norm=float(residual_norm), normal_residual_norm=float(normal_norm),
        operator_norm_estimate=float(operator_norm), condition_estimate=float(condition),
        update_norm=float(norm), nominal_opening_amplitude_increment_mm=(
            basis.physical_amplitudes(reduced@full.features[nominal])*1e3).tolist())
    return basis.expand(reduced), full, details
