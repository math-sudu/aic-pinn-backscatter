"""Evaluate measured boundary histories and conditional observation transfer."""

import json
import time

import numpy as np
import torch

from analyze_asymmetric_boundary import FullEquilibriumSolver, load_problem
from boundary_history import ForceHistoryProblem, finite_arm_probes, support_pressure_load, support_quadrature
from calibrate_support_response import affine_fit, json_values, loading_window, local_invariants, measured_crossing
from initial_state_model import relative_edge_probes, solve_intact
from stateful_physics import ROOT


def measured_region(history, time_s):
    times = history["time_load_s"]
    i = np.searchsorted(times, time_s)-1
    if i < 0 or i+1 >= len(times) or times[i+1]-times[i] > 1.5*np.median(np.diff(times)):
        raise ValueError("Regional probe is outside adjacent acquired frames")
    fraction = (time_s-times[i])/(times[i+1]-times[i])
    value = ((1-fraction)*history["q_dofs_microstrain"][i]
             + fraction*history["q_dofs_microstrain"][i+1])
    if not np.isfinite(value).all():
        raise ValueError("Regional probe has incomplete optical support")
    return value


def to_layers(q):
    return np.stack([q[..., 0]-q[..., 1], q[..., 0]+q[..., 1]], axis=-1)


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    config_path = ROOT / "inputs/boundary_history.json"
    config = json.loads(config_path.read_text())
    asymmetric_config = json.loads((ROOT/config["asymmetric_configuration"]).read_text())
    asymmetric = json.loads((ROOT/config["asymmetric_report"]).read_text())
    source = next(row for row in asymmetric["cases"] if row["name"] == config["reference_case"])
    saved = dict(np.load(ROOT/source["arrays"]))
    p, _, _ = load_problem(asymmetric_config)
    p.anchor_reference = p.tensor(saved["anchor_reference_m"])
    p.anchor_stiffness = p.tensor(saved["anchor_stiffness_n_per_m"])
    p.jack_operator = p.tensor(saved["jack_operator"])
    reference = saved["displacement_m"][1]
    reference_force = saved["force_n"][1]
    reference_rods = saved["rod_force_n"][1]
    solver = FullEquilibriumSolver(p, saved["displacement_m"][4])
    points, area = support_quadrature(p)
    gap = np.einsum("qa,fa->fq", p.shape.numpy(), reference.reshape(-1, 3)[p.mesh["trestle_faces"], 2])
    reference_pressure = -np.minimum(gap, 0)*p.settings["geometry"]["trestle_stiffness_pa_per_m"]
    priors = dict(uniform=area, reference_contact=area*reference_pressure)
    replay = ForceHistoryProblem(p, reference, asymmetric_config["conditional_rod_stiffness_n_per_m"])
    history = dict(np.load(ROOT/asymmetric_config["history_file"]))
    mapping = np.array(asymmetric_config["channels_at_model_corners"])-1
    centres = p.anchor_operator.numpy()[:, 2::3] @ p.mesh["xyz"][:, :2]
    gravity_moment = p.gravity.numpy()[2::3] @ p.mesh["xyz"][:, :2]
    eccentricity = saved["eccentricity_m"]
    lengths = np.diff(p.settings["observations"]["regions_m"], axis=1).ravel()
    azimuths = np.arange(0., 120., config["azimuth_step_deg"])
    # Every angle is constructed once and reused for both pressure cases.
    angles = np.concatenate([azimuths, azimuths+120., azimuths+240.])
    edge_operator, _ = relative_edge_probes(p.mesh, p.inverse, angles)
    arm_operator, _, _ = finite_arm_probes(p, angles, config["lvdt_arm_length_m"])
    print(json.dumps(dict(event="operators_ready", elapsed_s=time.perf_counter()-started)), flush=True)
    output = ROOT / "results/ideation/B2_boundary_history.json"
    destination = ROOT / "data/processed/boundary_history"
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(configuration=config_path.relative_to(ROOT).as_posix(),
        source_states=source["arrays"], source_history=asymmetric_config["history_file"],
        channels_at_model_corners=asymmetric_config["channels_at_model_corners"],
        reference_force_n=reference_force, reference_rods_n=reference_rods,
        eccentricity_m=eccentricity, rod_windows=[], cases=[], continuation=config["continuation"])
    diagonal = np.prod(np.sign(centres), axis=1)
    for window in config["rod_slope_windows_kn"]:
        selected = loading_window(history, window)
        force = history["force_kn"][selected]
        rods = history["rod_force_increment_kn"][selected][:, mapping]
        coefficients, rms = affine_fit(force, rods)
        report["rod_windows"].append(dict(force_window_kn=window, samples=int(selected.sum()),
            rod_slopes_kn_per_kn=coefficients[1], rod_fit_rms_kn=rms,
            support_slope_kn_per_kn=coefficients[1].sum()-1,
            diagonal_slope_kn_per_kn=coefficients[1]@diagonal/4))
    stages = [measured_crossing(history, level) for level in config["force_levels_kn"]]
    forces = np.r_[reference_force, np.array(config["force_levels_kn"])*1000]
    rod_forces = np.vstack([reference_rods,
        np.array([row["rod_force_increment_kn"][mapping] for row in stages])*1000 + reference_rods])
    for name in config["support_pressure_priors"]:
        states, support_forces, support_vectors, iterations = [], [], [], []
        for force, rods in zip(forces, rod_forces):
            total = -p.gravity.numpy().sum()+rods.sum()-force
            moment = rods@centres-force*eccentricity-gravity_moment
            vector, quadrature_force, _ = support_pressure_load(p, total, moment, priors[name])
            replay.set_boundary(rods, vector)
            state, updates = solve_intact(replay, force, states[-1] if states else reference, linear_solver=solver)
            states.append(state)
            support_forces.append(quadrature_force)
            support_vectors.append(vector)
            iterations.append(updates)
            print(json.dumps(dict(case=name, force_kn=float(force/1000), updates=updates,
                                  elapsed_s=time.perf_counter()-started)), flush=True)
        states = np.stack(states)
        support_forces, support_vectors = np.stack(support_forces), np.stack(support_vectors)
        replay.set_boundary(rod_forces, support_vectors)
        residual, damage, _ = replay.equilibrium(p.tensor(states)[None], p.tensor(forces)[None])
        delta = states-states[0]
        host = p.observations(p.tensor(delta))[0].numpy()*1e6
        edge = (delta@edge_operator.T).reshape(len(forces), 3, len(azimuths)).transpose(0, 2, 1)*1000
        arm = (delta@arm_operator.T).reshape(len(forces), 3, len(azimuths)).transpose(0, 2, 1)*1000
        rows = []
        for i, record in enumerate(stages, start=1):
            level = config["force_levels_kn"][i-1]
            observed_q = measured_region(history, record["time_load_s"])
            old_i = np.flatnonzero(saved["force_n"] == level*1000)[0]
            row = dict(force_kn=level, time_load_s=record["time_load_s"],
                support_force_kn=support_forces[i].sum()/1000,
                support_centroid_m=np.einsum("fq,fqc->c", support_forces[i], points[..., :2])/support_forces[i].sum(),
                rod_force_increment_kn=(rod_forces[i]-rod_forces[0])/1000,
                host_q_microstrain=host[i], observed_fibre_q_microstrain=observed_q,
                host_change_from_equal_anchor_microstrain=host[i]-saved["host_increment_microstrain"][old_i],
                fibre_minus_host_layer_extension_mm=(to_layers(observed_q)-to_layers(host[i]))*lengths[None, :, None]*1e-3,
                measured_lvdt_mm=record["cone_relative_lvdt_increment_mm"],
                measured_local_mean_amplitude_mm=local_invariants(record["cone_relative_lvdt_increment_mm"]))
            for label, values in (("edge", edge[i]), ("finite_arm", arm[i])):
                invariant = local_invariants(values)
                row[label+"_mean_range_mm"] = [invariant[:, 0].min(), invariant[:, 0].max()]
                row[label+"_amplitude_range_mm"] = [invariant[:, 1].min(), invariant[:, 1].max()]
            rows.append(row)
        file = destination/f"{name}.npz"
        np.savez_compressed(file, force_n=forces, displacement_m=states, rod_force_n=rod_forces,
            support_quadrature_force_n=support_forces, support_load_n=support_vectors,
            host_increment_microstrain=host, azimuth_deg=azimuths,
            edge_increment_mm=edge, finite_arm_increment_mm=arm,
            gauge_reference=replay.gauge_reference.numpy())
        report["cases"].append(dict(name=name, arrays=file.relative_to(ROOT).as_posix(), stages=rows,
            maximum_nodal_residual_n=float(residual.abs().max()), maximum_damage=float(damage.max()),
            maximum_gauge_reaction_n=float(replay.gauge_reactions(p.tensor(states)).abs().max()),
            newton_updates=iterations))
        report["elapsed_s"] = time.perf_counter()-started
        output.write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    a, b = [dict(np.load(ROOT/case["arrays"])) for case in report["cases"]]
    report["support_shape_effect"] = dict(
        maximum_host_q_difference_microstrain=float(np.max(np.abs(a["host_increment_microstrain"]-b["host_increment_microstrain"]))),
        maximum_edge_difference_mm=float(np.max(np.abs(a["edge_increment_mm"]-b["edge_increment_mm"]))),
        maximum_finite_arm_difference_mm=float(np.max(np.abs(a["finite_arm_increment_mm"]-b["finite_arm_increment_mm"]))))
    report["transfer_interpretation"] = (
        "Fibre-minus-host regional extensions are the correction required by this conditional bulk/interface model and coordinate convention. "
        "They combine model/material/registration differences and cable endpoint transfer; they are not identified slip. "
        "Finite-arm values are a rigid-fixture sensitivity at a declared span, without fitting instrument geometry.")
    output.write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
