"""Conditional force-history boundaries and objective local-motion probes."""

import numpy as np
from scipy import sparse
from scipy.optimize import root

from initial_state_model import point_value, relative_edge_probes
from quadratic_elements import shape_gradients


def support_quadrature(p):
    faces = p.mesh["trestle_faces"]
    shape, weights = p.shape.numpy(), p.surface_weights.numpy()
    points = np.einsum("qa,fac->fqc", shape, p.mesh["xyz"][faces])
    area = p.mesh["trestle_face_area"][:, None] * weights
    return points, area


def support_pressure_load(p, resultant_n, first_moment_n_m, prior):
    """Nonnegative quadrature loads with prescribed vertical force and moment."""
    points, area = support_quadrature(p)
    if resultant_n <= 0:
        raise ValueError("Force replay requires a positive conditional support resultant")
    base = np.asarray(prior, dtype=float).ravel()
    if base.shape != (area.size,) or np.any(base < 0) or not np.any(base > 0):
        raise ValueError("The support prior must be nonnegative quadrature-force weights")
    xy = points[..., :2].reshape(-1, 2)
    centre = np.asarray(first_moment_n_m) / resultant_n

    def distribution(tilt):
        log_weight = np.full_like(base, -np.inf)
        positive = base > 0
        log_weight[positive] = np.log(base[positive]) + xy[positive] @ tilt
        value = np.exp(log_weight - log_weight.max())
        return value / value.sum()

    def moments(tilt):
        return distribution(tilt) @ xy - centre

    def derivative(tilt):
        probability = distribution(tilt)
        centred = xy - probability @ xy
        return centred.T @ (probability[:, None] * centred)

    fit = root(moments, np.zeros(2), jac=derivative)
    if not np.isfinite(fit.x).all() or np.max(np.abs(moments(fit.x))) > 1e-9:
        raise ValueError("Support force and moment cannot be represented by this pressure prior")
    force = (resultant_n * distribution(fit.x)).reshape(area.shape)
    local = np.einsum("fq,qa->fa", force, p.shape.numpy())
    vector = np.zeros(p.ndof)
    np.add.at(vector, 3*p.mesh["trestle_faces"].ravel()+2, local.ravel())
    return vector, force, force / area


class ForceHistoryProblem:
    """Known force tractions, original material history and a vertical rigid gauge.

    The three gauge coordinates fix common vertical translation and tilt. Their
    reactions vanish for equilibrated force/moment inputs; the four measured
    rod forces are external loads, not springs or displacement labels.
    """

    def __init__(self, p, reference_state, gauge_stiffness_n_per_m):
        self.p, self.ndof, self.tensor = p, p.ndof, p.tensor
        centres = p.anchor_operator.numpy()[:, 2::3] @ p.mesh["xyz"][:, :2]
        modes = np.column_stack([np.ones(4), centres])
        modes /= np.linalg.norm(modes, axis=0)
        self.gauge = p.tensor(modes.T @ p.anchor_operator.numpy())
        self.gauge_reference = self.gauge @ p.tensor(reference_state)
        self.gauge_stiffness = gauge_stiffness_n_per_m
        operator = sparse.csr_matrix(self.gauge.numpy())
        self.gauge_matrix = gauge_stiffness_n_per_m * (operator.T @ operator)
        self.rod_force = self.support_load = None
        p.settings["geometry"]["trestle_stiffness_pa_per_m"] = 0.

    def set_boundary(self, rod_force_n, support_load_n):
        self.rod_force = self.tensor(rod_force_n)
        self.support_load = self.tensor(support_load_n)

    def gauge_reactions(self, displacement):
        return self.gauge_stiffness * (displacement @ self.gauge.T - self.gauge_reference)

    def equilibrium(self, displacement, jack_force_n, compression_stiffness=None):
        force, damage, history = self.p.forces(displacement, compression_stiffness=compression_stiffness)
        external = (self.p.gravity + jack_force_n[..., None]*self.p.jack_operator
                    + self.support_load - self.rod_force @ self.p.anchor_operator)
        residual = force - external + self.gauge_reactions(displacement) @ self.gauge
        return residual, damage, history

    def intact_tangent(self, displacement):
        return (self.p.intact_bulk_interface + self.p.interface_closure_tangent(displacement)
                + self.p.wall_tangent(displacement)
                + self.gauge_matrix).tocsc()

    def history_tangent(self, displacement, prior_history):
        return (self.p.intact_bulk_interface
                + self.p.interface_history_correction_tangent(displacement, prior_history)
                + self.p.wall_tangent(displacement) + self.gauge_matrix).tocsc()


def displacement_gradient(p, body, point):
    """Operator grad(u)[component, derivative] at one point of a named body."""
    elements = np.flatnonzero(p.mesh["body"] == body)
    bary = np.einsum("i,eij->ej", np.r_[1, point], p.inverse[elements])
    valid = np.flatnonzero(np.min(bary, axis=1) >= -1e-9)
    if not len(valid):
        raise ValueError("The observation attachment lies outside its body")
    local = valid[0]
    element = elements[local]
    gradient = p.inverse[element, 1:, :].T
    if p.mesh["tetra"].shape[1] == 10:
        gradient = shape_gradients(bary[local], gradient)
    result = np.zeros((3, 3, p.ndof))
    for component in range(3):
        result[component][:, 3*p.mesh["tetra"][element]+component] = gradient.T
    return result


def finite_arm_probes(p, angles_deg, arm_length_m):
    """Cone-edge displacement relative to a co-rotating substrate fixture.

    u_c(c) - u_s(s) - omega_s(s) x (c-s), projected on the vertical axis.
    The attachment is radially outward from the cone edge by arm_length_m.
    This idealization supplies a conditional observation, not actual LVDT geometry.
    """
    _, tips = relative_edge_probes(p.mesh, p.inverse, angles_deg)
    operators, attachments = [], []
    for tip in tips:
        ray = tip[:2] / np.linalg.norm(tip[:2])
        attachment = tip + np.r_[arm_length_m*ray, 0.]
        span = tip-attachment
        gradient = displacement_gradient(p, 1, attachment)
        omega_x = .5*(gradient[2, 1]-gradient[1, 2])
        omega_y = .5*(gradient[0, 2]-gradient[2, 0])
        operator = (point_value(p.mesh, p.inverse, 2, tip)
                    - point_value(p.mesh, p.inverse, 1, attachment)
                    - span[1]*omega_x + span[0]*omega_y)
        operators.append(operator)
        attachments.append(attachment)
    return np.asarray(operators), tips, np.asarray(attachments)
