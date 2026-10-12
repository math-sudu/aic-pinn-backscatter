"""Calibrate observable support release and conditional machine transfer.

Only total rod-force slope and external displacement transfer are fitted.
Individual rod increments and local relative-motion invariants are checks.
The intact solves are confined to the stated early loading interval and reject
cohesive damage through the existing initializer.
"""

import json
import time

import numpy as np
from scipy.optimize import root_scalar
import torch

from initial_state_model import InitialStatePhysics, material_densities, solve_intact
from spatial_resolution import symmetric_projection
from stateful_physics import ROOT


def affine_fit(x, values):
    """Return [offset, slope] and residual RMS, preserving output columns."""
    x, values = np.asarray(x), np.asarray(values)
    matrix = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(matrix, values, rcond=None)[0]
    return coefficients, np.sqrt(np.mean((matrix @ coefficients - values) ** 2, axis=0))


def local_invariants(displacements):
    """Equal-angle mean and harmonic amplitude, independent of channel order."""
    values = np.asarray(displacements)
    mean = values.mean(axis=-1)
    amplitude = np.sqrt(2 * np.mean((values - mean[..., None]) ** 2, axis=-1))
    return np.stack([mean, amplitude], axis=-1)


def loading_window(history, force_window):
    force = history["force_kn"]
    return ((np.arange(len(force)) < force.argmax())
            & (force >= force_window[0]) & (force <= force_window[1]))


def measured_crossing(history, level):
    """Interpolate the first rising crossing, checking its acquisition interval."""
    force, time_s = history["force_kn"], history["time_load_s"]
    i = np.flatnonzero((force[:-1] < level) & (force[1:] >= level)
                      & (np.arange(len(force) - 1) < force.argmax()))[0]
    if time_s[i + 1] - time_s[i] > 1.5 * np.median(np.diff(time_s)):
        raise ValueError("The requested force crossing spans an acquisition gap")
    alpha = (level - force[i]) / (force[i + 1] - force[i])
    return {name: (1 - alpha) * history[name][i] + alpha * history[name][i + 1]
            for name in ("time_load_s", "force_kn", "force_increment_kn", "jack_increment_mm",
                         "rod_force_increment_kn", "cone_relative_lvdt_increment_mm",
                         "inferred_trestle_force_change_kn")}


def fit_support(p, projection, stiffness, reference_force, force_window, target_slope):
    """Fit contact compliance conditional on a declared common rod stiffness."""
    p.anchor_stiffness = p.tensor(np.full(4, stiffness))
    area = p.mesh["trestle_face_area"].sum()
    guess = 4 * stiffness * (1 / target_slope - 1) / area
    forces = np.r_[0., reference_force, np.asarray(force_window) * 1000]
    previous = None

    def evaluate(log_stiffness):
        nonlocal previous
        p.settings["geometry"]["trestle_stiffness_pa_per_m"] = float(np.exp(log_stiffness))
        p.anchor_reference = None
        states = []
        for j, force in enumerate(forces):
            state, _ = solve_intact(p, force, previous[j] if previous is not None
                                   else (None if not states else states[-1]), projection)
            if j == 0:
                p.seat_anchors(p.tensor(state))
            states.append(state)
        previous = np.stack(states)
        outputs = p.mechanical_outputs(p.tensor(previous))
        rods = outputs["rod_force_n"].numpy().sum(axis=-1)
        slope = (rods[-1] - rods[-2]) / (forces[-1] - forces[-2])
        row = dict(trestle_stiffness_pa_per_m=float(np.exp(log_stiffness)),
                   total_rod_slope=float(slope))
        print(json.dumps(dict(rod_stiffness_n_per_m=stiffness, **row)), flush=True)
        return slope - target_slope

    result = root_scalar(evaluate, method="secant", x0=np.log(guess), x1=np.log(guess) + .15,
                         xtol=1e-4)
    if not result.converged:
        raise RuntimeError("Conditional support calibration did not converge")
    # Ensure the retained state uses the reported root, not the previous secant point.
    evaluate(result.root)
    return forces, previous


def json_values(value):
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    config_path = ROOT / "inputs/support_calibration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads((ROOT / config["initial_model_report"]).read_text(encoding="utf-8"))
    initial_config = json.loads((ROOT / config["initial_model_configuration"]).read_text(encoding="utf-8"))
    history = dict(np.load(ROOT / config["history_file"]))
    mesh = dict(np.load(ROOT / report["mesh_file"]))
    p = InitialStatePhysics(mesh, report["resolved_settings"], initial_config,
                            material_densities(initial_config["weight"])[0])
    projection = symmetric_projection(p)
    mask = loading_window(history, config["rod_fit_force_window_kn"])
    coefficients, rod_rms = affine_fit(history["force_increment_kn"][mask],
                                     history["rod_force_increment_kn"][mask])
    target_slope = coefficients[1].sum()
    destination = ROOT / "data/processed/support_calibration"
    destination.mkdir(parents=True, exist_ok=True)
    output = ROOT / "results/ideation/B2_support_calibration.json"
    result = dict(configuration=config_path.relative_to(ROOT).as_posix(),
                  source_history=config["history_file"], fit_samples=int(mask.sum()),
                  source_basis=[
                      "references/papers/jansson2024_dataset/full.md section 4.3: four rods, two trestles, three local LVDTs and separate jack displacement.",
                      "references/papers/jansson2025_lining/full.md sections 2.2-2.3: assumed zero pretension after hand tightening and 120-degree LVDT spacing; absolute instrument registration is not specified.",
                      "https://pmc.ncbi.nlm.nih.gov/articles/PMC11460454/: dataset paper checked again for instrument definitions; actual DAT column names determine parsing."
                  ],
                  rod_fit_coefficients=coefficients, rod_fit_rms_kn=rod_rms,
                  total_rod_fit_slope=target_slope,
                  fit_definition=config["rod_fit_reason"],
                  support_law_interpretation="The fitted contact-bed stiffness represents effective trestle/bearing/seating compliance conditional on rod stiffness and the intact solid/interface scenario; it is not a penalty selected for numerical convergence or a separately measured fixture property.",
                  reference_force_n=report["known_load_reference_force_n"],
                  cases=[], continuation=config["continuation"])
    for stiffness in config["conditional_rod_stiffness_n_per_m"]:
        forces, states = fit_support(p, projection, stiffness,
            report["known_load_reference_force_n"], config["rod_fit_force_window_kn"], target_slope)
        force_list, state_list = list(forces), list(states)
        for force in np.asarray(config["force_samples_kn"]) * 1000:
            if force in force_list:
                continue
            state, _ = solve_intact(p, force, state_list[-1], projection)
            force_list.append(force)
            state_list.append(state)
        order = np.argsort(force_list)
        forces = np.asarray(force_list)[order]
        states = np.stack(state_list)[order]
        tensor = p.tensor(states)[None]
        residual, damage, _ = p.equilibrium(tensor, p.tensor(forces)[None])
        mechanical = {name: value.numpy() for name, value in p.mechanical_outputs(p.tensor(states)).items()}
        rod_delta = (mechanical["rod_force_n"] - mechanical["rod_force_n"][1]) / 1000
        local_delta = (mechanical["local_relative_m"] - mechanical["local_relative_m"][1]) * 1000
        jack_delta = (mechanical["jack_face_mean_m"] - mechanical["jack_face_mean_m"][1]) * 1000
        trestle_delta = (mechanical["trestle_force_n"] - mechanical["trestle_force_n"][1]) / 1000
        fit_mask = loading_window(history, config["external_displacement_fit_force_window_kn"])
        computed_jack = np.interp(history["force_kn"], forces / 1000, jack_delta)
        difference = history["jack_increment_mm"] - computed_jack
        transfer, transfer_rms = affine_fit(history["force_increment_kn"][fit_mask], difference[fit_mask])
        predicted_machine = computed_jack + transfer[0] + transfer[1] * history["force_increment_kn"]
        check_mask = loading_window(history, config["external_displacement_check_force_window_kn"])
        stages = []
        for level in config["force_samples_kn"]:
            i = int(np.flatnonzero(forces == level * 1000)[0])
            observed = measured_crossing(history, level)
            stages.append(dict(force_kn=level, time_load_s=observed["time_load_s"],
                measured_rod_increment_kn=observed["rod_force_increment_kn"],
                model_rod_increment_kn=rod_delta[i],
                measured_trestle_change_kn=observed["inferred_trestle_force_change_kn"],
                model_trestle_change_kn=trestle_delta[i],
                measured_local_mean_amplitude_mm=local_invariants(observed["cone_relative_lvdt_increment_mm"]),
                model_local_mean_amplitude_mm=local_invariants(local_delta[i]),
                measured_jack_increment_mm=observed["jack_increment_mm"],
                model_jack_face_increment_mm=jack_delta[i],
                transferred_machine_increment_mm=jack_delta[i] + transfer[0]
                    + transfer[1] * observed["force_increment_kn"]))
        file = destination / f"rods_{stiffness / 1e6:g}MN.npz"
        host, _ = p.observations(tensor)
        np.savez_compressed(file, force_n=forces, displacement_m=states,
            anchor_reference_m=p.anchor_reference.numpy(),
            rod_force_increment_kn=rod_delta, local_relative_increment_mm=local_delta,
            jack_face_increment_mm=jack_delta, trestle_force_change_kn=trestle_delta,
            host_increment_microstrain=(host[0] - host[0, 1]).numpy() * 1e6,
            **mechanical)
        balance = forces + mechanical["trestle_force_n"] - mechanical["rod_force_n"].sum(axis=-1) + float(p.gravity.sum())
        case = dict(rod_stiffness_n_per_m=stiffness,
            trestle_stiffness_pa_per_m=p.settings["geometry"]["trestle_stiffness_pa_per_m"],
            arrays=file.relative_to(ROOT).as_posix(),
            external_displacement_offset_mm=transfer[0],
            external_displacement_slope_mm_per_kn=transfer[1],
            external_displacement_fit_rms_mm=transfer_rms,
            external_displacement_check_rms_mm=float(np.sqrt(np.mean(
                (predicted_machine[check_mask] - history["jack_increment_mm"][check_mask]) ** 2))),
            maximum_damage=float(damage.max()), maximum_nodal_residual_n=float(residual.abs().max()),
            maximum_vertical_balance_n=float(abs(balance).max()), stages=stages)
        result["cases"].append(case)
        result["elapsed_s"] = time.perf_counter() - started
        output.write_text(json.dumps(json_values(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(dict(completed_case=stiffness, elapsed_s=result["elapsed_s"])), flush=True)
    first, second = [np.load(ROOT / row["arrays"]) for row in result["cases"]]
    # Positive support release and local response, not arbitrary fit precision,
    # determine what these conditional scenarios can explain.
    result["conditional_comparison"] = dict(
        rod_stiffness_ratio=config["conditional_rod_stiffness_n_per_m"][1]
            / config["conditional_rod_stiffness_n_per_m"][0],
        maximum_rod_increment_difference_kn=float(np.max(abs(
            second["rod_force_increment_kn"] - first["rod_force_increment_kn"]))),
        maximum_local_increment_difference_mm=float(np.max(abs(
            second["local_relative_increment_mm"] - first["local_relative_increment_mm"]))),
        maximum_host_increment_difference_microstrain=float(np.max(abs(
            second["host_increment_microstrain"] - first["host_increment_microstrain"]))),
        external_displacement_slope_change_mm_per_kn=result["cases"][1]["external_displacement_slope_mm_per_kn"]
            - result["cases"][0]["external_displacement_slope_mm_per_kn"],
        interpretation="These conditional fits test observability of support parameters. A comparable machine fit with different effective-rod stiffness is not an identified rod modulus/length. Individual rod allocation and local mean/harmonic response remain independent physical checks."
    )
    output.write_text(json.dumps(json_values(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
