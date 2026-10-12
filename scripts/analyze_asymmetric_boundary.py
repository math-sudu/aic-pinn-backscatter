"""Conditional eccentric loading and diagonal anchor compliance.

All displacement degrees of freedom remain active. Reflection factors only
precondition the full contact equilibrium; they do not impose a symmetry.
"""

import copy
import itertools
import json
import time

import numpy as np
from scipy.optimize import root, root_scalar
from scipy.sparse.linalg import LinearOperator, cg, splu
import torch

from analyze_local_boundary import azimuth_correspondence, json_values
from calibrate_support_response import local_invariants, measured_crossing
from initial_state_model import InitialStatePhysics, material_densities, solve_intact, surface_faces
from quadratic_elements import shape_values, triangle_rule
from spatial_resolution import reflection_projection
from stateful_physics import ROOT


def eccentric_patch(mesh, body, level, radius, eccentricity):
    """Positive quadrature pressure with a prescribed resultant and first moment."""
    faces, area = surface_faces(mesh, body, level)
    selected = np.linalg.norm(mesh["xyz"][faces[:, :3], :2].mean(axis=1), axis=1) < radius
    faces, area = faces[selected], area[selected]
    bary, weight = triangle_rule()
    shape = bary if faces.shape[1] == 3 else shape_values(bary)
    points = np.einsum("qa,fac->fqc", bary, mesh["xyz"][faces[:, :3], :2])
    scaled = points.reshape(-1, 2) / radius
    measure = (area[:, None] * weight).ravel()
    target = np.asarray(eccentricity) / radius

    def distribution(tilt):
        exponent = scaled @ tilt
        probability = measure * np.exp(exponent - exponent.max())
        return probability / probability.sum()

    def moments(tilt):
        return distribution(tilt) @ scaled - target

    def jacobian(tilt):
        probability = distribution(tilt)
        centred = scaled - probability @ scaled
        return centred.T @ (probability[:, None] * centred)

    fit = root(moments, np.zeros(2), jac=jacobian)
    # A centred symmetric patch is already solved at roundoff. HYBR may
    # report no progress there, so check the defining moment equations.
    mismatch = moments(fit.x)
    if not np.isfinite(mismatch).all() or np.max(np.abs(mismatch)) > 1e-10:
        raise ValueError("The requested eccentricity has no resolved positive pressure on this footprint")
    probability = distribution(fit.x).reshape(points.shape[:2])
    nodal = np.einsum("fq,qa->fa", probability, shape)
    operator = np.zeros(3 * len(mesh["xyz"]))
    np.add.at(operator, 3 * faces.ravel() + 2, nodal.ravel())
    return operator, dict(eccentricity_m=probability.ravel() @ points.reshape(-1, 2),
        dimensionless_pressure_tilt=fit.x, pressure_per_unit_force_pa_per_n=probability / measure.reshape(probability.shape),
        quadrature_points_m=points, quadrature_force_fraction=probability)


class FullEquilibriumSolver:
    """Use four symmetric reference factors for coupled asymmetric tangents."""

    def __init__(self, p, reference_state):
        matrix = p.intact_tangent(reference_state)
        self.parities = list(itertools.product((1, -1), repeat=2))
        self.projections = [reflection_projection(p, parity) for parity in self.parities]
        self.factors = [splu((q.T @ matrix @ q).tocsc(), permc_spec="MMD_AT_PLUS_A")
                        for q in self.projections]
        self.preconditioner = LinearOperator(matrix.shape, matvec=self.precondition, dtype=float)

    def precondition(self, value):
        return sum(q @ factor.solve(q.T @ value) for q, factor in zip(self.projections, self.factors))

    def __call__(self, matrix, rhs):
        result, status = cg(matrix, rhs, M=self.preconditioner, rtol=1e-10, atol=1e-7)
        if status:
            raise RuntimeError(f"Full asymmetric equilibrium linear solve failed: {status}")
        return result


def load_problem(config):
    support = json.loads((ROOT / config["support_report"]).read_text())
    selected = next(row for row in support["cases"]
                    if row["rod_stiffness_n_per_m"] == config["conditional_rod_stiffness_n_per_m"])
    settings = json.loads((ROOT / support["configuration"]).read_text())
    initial = json.loads((ROOT / settings["initial_model_report"]).read_text())
    model = json.loads((ROOT / settings["initial_model_configuration"]).read_text())
    resolved = copy.deepcopy(initial["resolved_settings"])
    resolved["geometry"]["trestle_stiffness_pa_per_m"] = selected["trestle_stiffness_pa_per_m"]
    mesh = dict(np.load(ROOT / initial["mesh_file"]))
    p = InitialStatePhysics(mesh, resolved, model, material_densities(model["weight"])[0])
    p.anchor_stiffness = p.tensor(np.full(4, selected["rod_stiffness_n_per_m"]))
    saved = dict(np.load(ROOT / selected["arrays"]))
    p.anchor_reference = p.tensor(saved["anchor_reference_m"])
    return p, saved, selected


def trestle_resultants(p, states):
    """Vertical resultant and first moment from the same contact quadrature."""
    conn = p.mesh["trestle_faces"]
    shape, weights = p.shape.numpy(), p.surface_weights.numpy()
    gap = np.einsum("qa,sfa->sfq", shape, states.reshape(len(states), -1, 3)[:, conn, 2])
    force = -np.minimum(gap, 0) * p.settings["geometry"]["trestle_stiffness_pa_per_m"]
    force *= p.mesh["trestle_face_area"][None, :, None] * weights
    xy = np.einsum("qa,fac->fqc", shape, p.mesh["xyz"][conn, :2])
    return force.sum(axis=(1, 2)), np.einsum("sfq,fqc->sc", force, xy)


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    config_path = ROOT / "inputs/asymmetric_boundary_response.json"
    config = json.loads(config_path.read_text())
    boundary = json.loads((ROOT / config["boundary_report"]).read_text())
    mapping = next(row for row in boundary["moment_admissibility"]["mappings"]
                   if row["channels_at_model_corners"] == config["channels_at_model_corners"])
    eccentricity = np.mean(mapping["fixed_eccentricity_polygon_m"], axis=0)
    p, saved, support_case = load_problem(config)
    geometry = p.settings["geometry"]
    operator, pressure = eccentric_patch(p.mesh, 2, geometry["bottom_levels_m"][-1],
                                         geometry["jack_patch_radius_m"], eccentricity)
    p.jack_operator = p.tensor(operator)
    reference_state = saved["displacement_m"][np.flatnonzero(saved["force_n"] == 25000)[0]]
    solver = FullEquilibriumSolver(p, reference_state)
    print(json.dumps(dict(solver_ready_s=time.perf_counter()-started, eccentricity_m=eccentricity.tolist())), flush=True)
    history = dict(np.load(ROOT / config["history_file"]))
    indices = np.array(config["channels_at_model_corners"]) - 1
    centres = np.array(geometry["bolt_centres_m"])
    diagonal = np.prod(np.sign(centres), axis=1)
    levels = np.array(config["anchor_fit_force_levels_kn"])
    measured = [measured_crossing(history, level)["rod_force_increment_kn"][indices] for level in levels]
    target = ((measured[1] - measured[0]) @ diagonal / 4) / np.diff(levels)[0]
    previous = [saved["displacement_m"][np.flatnonzero(saved["force_n"] == level*1000)[0]] for level in levels]

    def fit_anchor(eta):
        p.anchor_stiffness = p.tensor(config["conditional_rod_stiffness_n_per_m"] * np.exp(eta * diagonal))
        for i, level in enumerate(levels):
            previous[i], _ = solve_intact(p, level * 1000, previous[i], linear_solver=solver)
        rods = p.anchor_loads(p.tensor(np.stack(previous))).numpy() / 1000
        slope = (rods[1] - rods[0]) @ diagonal / 4 / np.diff(levels)[0]
        print(json.dumps(dict(anchor_log_contrast=float(eta), diagonal_slope=float(slope),
                              elapsed_s=time.perf_counter()-started)), flush=True)
        return slope - target

    destination = ROOT / "data/processed/asymmetric_boundary_response"
    destination.mkdir(parents=True, exist_ok=True)
    output = ROOT / "results/ideation/B2_asymmetric_boundary_response.json"
    report = dict(configuration=config_path.relative_to(ROOT).as_posix(),
        source_states=support_case["arrays"], source_history=config["history_file"],
        channels_at_model_corners=config["channels_at_model_corners"],
        eccentricity_m=eccentricity, pressure={key: value for key, value in pressure.items()
            if key not in ("quadrature_points_m", "quadrature_force_fraction", "pressure_per_unit_force_pa_per_n")},
        pressure_range_per_unit_force_pa_per_n=[pressure["pressure_per_unit_force_pa_per_n"].min(),
                                               pressure["pressure_per_unit_force_pa_per_n"].max()],
        anchor_fit_target_diagonal_slope=target, cases=[], continuation=config["continuation"])
    forces = np.r_[saved["force_n"][1], np.array(config["force_levels_kn"])*1000]

    def run_case(name, eta):
        p.anchor_stiffness = p.tensor(config["conditional_rod_stiffness_n_per_m"] * np.exp(eta * diagonal))
        states, iterations = [saved["displacement_m"][0]], [0]
        for force in forces:
            state, count = solve_intact(p, force, states[-1], linear_solver=solver)
            states.append(state)
            iterations.append(count)
            print(json.dumps(dict(case=name, force_kn=float(force/1000), updates=count,
                                  elapsed_s=time.perf_counter()-started)), flush=True)
        states, case_forces = np.stack(states), np.r_[0., forces]
        tensor = p.tensor(states)
        residual, damage, _ = p.equilibrium(tensor[None], p.tensor(case_forces)[None])
        mechanical = {key: value.numpy() for key, value in p.mechanical_outputs(tensor).items()}
        host, _ = p.observations(tensor)
        host_delta = (host - host[1]).numpy() * 1e6
        rod_delta = (mechanical["rod_force_n"] - mechanical["rod_force_n"][1]) / 1000
        local_delta = (mechanical["local_relative_m"] - mechanical["local_relative_m"][1]) * 1000
        angles, probes, azimuth = azimuth_correspondence(p, states, case_forces, history, config["azimuth_step_deg"])
        reaction, moment = trestle_resultants(p, states)
        anchor_centres = p.anchor_operator.numpy()[:, 2::3] @ p.mesh["xyz"][:, :2]
        gravity_moment = p.gravity.numpy()[2::3] @ p.mesh["xyz"][:, :2]
        balance = case_forces + reaction - mechanical["rod_force_n"].sum(axis=1) + p.gravity.numpy().sum()
        moment_balance = case_forces[:, None]*eccentricity + moment - mechanical["rod_force_n"] @ anchor_centres + gravity_moment
        delta = states - states[1]
        components = [(q @ (q.T @ delta.T)).T for q in solver.projections]
        sector_host = np.stack([part @ p.target.numpy().T for part in components])
        sector_host = p.components(p.tensor(sector_host).reshape(4, len(states), 2, 3, 2)).numpy()*1e6
        stages = []
        for i, level in enumerate(config["force_levels_kn"], start=2):
            observed = measured_crossing(history, level)
            stages.append(dict(force_kn=level,
                measured_rod_increment_at_model_corners_kn=observed["rod_force_increment_kn"][indices],
                model_rod_increment_kn=rod_delta[i],
                measured_diagonal_rod_contrast_kn=observed["rod_force_increment_kn"][indices] @ diagonal / 4,
                model_diagonal_rod_contrast_kn=rod_delta[i] @ diagonal / 4,
                measured_local_mean_amplitude_mm=local_invariants(observed["cone_relative_lvdt_increment_mm"]),
                model_local_mean_amplitude_mm=local_invariants(local_delta[i]),
                azimuth=azimuth[i-2], trestle_force_kn=reaction[i]/1000,
                trestle_first_moment_kn_m=moment[i]/1000,
                host_q_microstrain=host_delta[i], host_right_minus_left_q_microstrain=host_delta[i, :, 2]-host_delta[i, :, 0]))
        file = destination / f"{name}.npz"
        np.savez_compressed(file, force_n=case_forces, displacement_m=states,
            anchor_reference_m=p.anchor_reference.numpy(), anchor_stiffness_n_per_m=p.anchor_stiffness.numpy(),
            jack_operator=operator, azimuth_deg=angles, azimuth_local_increment_mm=probes,
            host_increment_microstrain=host_delta, sector_host_increment_microstrain=sector_host,
            rod_force_increment_kn=rod_delta, local_relative_increment_mm=local_delta,
            trestle_first_moment_n_m=moment, **mechanical, **pressure)
        row = dict(name=name, anchor_log_contrast=eta, anchor_stiffness_n_per_m=p.anchor_stiffness.numpy(),
            arrays=file.relative_to(ROOT).as_posix(), stages=stages,
            maximum_nodal_residual_n=float(residual.abs().max()), maximum_damage=float(damage.max()),
            maximum_vertical_balance_n=np.abs(balance).max(), maximum_first_moment_balance_n_m=np.abs(moment_balance).max(),
            anchor_patch_centres_m=anchor_centres, parities=solver.parities,
            sector_reconstruction_max_m=np.max(np.abs(sum(components)-delta)),
            newton_updates=iterations)
        report["cases"].append(row)
        report["elapsed_s"] = time.perf_counter()-started
        output.write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")

    run_case("eccentric_equal_anchors", 0.)
    fit = root_scalar(fit_anchor, method="secant", x0=0., x1=-.3, xtol=1e-5)
    if not fit.converged:
        raise RuntimeError("The conditional anchor contrast fit did not converge")
    report["fitted_anchor_log_contrast"] = float(fit.root)
    run_case("eccentric_diagonal_anchors", float(fit.root))


if __name__ == "__main__":
    main()
