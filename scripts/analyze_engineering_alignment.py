"""Map the measured lining history to mechanical and regional sensing outputs.

The force balance uses increments from the first known-load reference. Regional
extensions integrate the measured cable strain; endpoint transfer to concrete
and the target installation remains a separate physical relation.
"""

import json

import numpy as np

from analyze_jansson_observations import ROOT, PROCESSED, interpolate_time, read_load


def regional_extensions(target, lengths):
    """Convert (q_m, q_b), in microstrain, to physical cable extensions in mm."""
    target = np.asarray(target, dtype=float)
    factor = np.asarray(lengths) * 1e-3
    mean = target[..., 0] * factor
    difference = 2 * target[..., 1] * factor
    layers = np.stack([mean - difference / 2, mean + difference / 2], axis=-1)
    return layers, mean, difference


def force_crossings(time, force, level):
    """First crossings before/after the global peak; do not cross acquisition gaps."""
    time, force = np.asarray(time), np.asarray(force)
    peak = int(force.argmax())
    index = np.arange(len(force) - 1)
    continuous = np.diff(time) < 1.5 * np.median(np.diff(time))
    conditions = {
        "rising_before_peak": (force[:-1] < level) & (force[1:] >= level) & (index < peak),
        "falling_after_peak": (force[:-1] > level) & (force[1:] <= level) & (index >= peak),
    }
    result = {}
    for name, condition in conditions.items():
        candidates = np.flatnonzero(condition & continuous)
        if len(candidates):
            j = candidates[0]
            result[name] = float(time[j] + (level - force[j]) / (force[j + 1] - force[j]) * (time[j + 1] - time[j]))
    return result


def mechanical_outputs(values, reference, channels):
    """Keep actuator and local displacement distinct; recover support increments."""
    index = {name: j for j, name in enumerate(channels)}
    delta = values - reference
    force = -values[..., index["Ch 1 Force"]]
    force_delta = -delta[..., index["Ch 1 Force"]]
    lvdt = delta[..., [index[f"LVDT{j}"] for j in (1, 2, 3)]]
    rods = delta[..., [index[f"Studforce {j}"] for j in (1, 2, 3, 4)]]
    return {
        "force_kn": force,
        "force_increment_kn": force_delta,
        "jack_increment_mm": delta[..., index["Ch 1 Displacement"]],
        "cone_relative_lvdt_increment_mm": lvdt,
        "cone_relative_mean_increment_mm": lvdt.mean(axis=-1),
        "cone_relative_lvdt_spread_mm": np.ptp(lvdt, axis=-1),
        "rod_force_increment_kn": rods,
        "inferred_trestle_force_change_kn": rods.sum(axis=-1) - force_delta,
    }


def cl_history(lengths, settings, alignment):
    _, _, frame = read_load(ROOT / settings["cl100h_alignment"]["load_file"])
    channels = alignment["aligned_records"][0]["channels"]
    raw_time = frame["Time"].to_numpy()
    raw_values = frame[channels].to_numpy()
    force = -frame["Ch 1 Force"].to_numpy()
    aligned = [dict(np.load(PROCESSED / f"CL100H-d{d}-load-aligned.npz")) for d in (1, 2)]
    reference = aligned[0]["known_load_reference_values"]
    # Use the acquired first-direction cadence. Direction 2 is interpolated
    # on the corrected shared clock, retaining incomplete support and gaps.
    first = aligned[0]
    known = first["known_load_mask"]
    time = first["time_load_s"][known]
    q = np.stack([
        interpolate_time(a["time_load_s"], a["target_from_known_load_reference_microstrain"].reshape(-1, 6), time).reshape(-1, 3, 2)
        for a in aligned
    ], axis=1)
    layers, mean, difference = regional_extensions(q, lengths)
    mechanical = mechanical_outputs(first["load_values"][known], reference, channels)
    destination = PROCESSED / "CL100H-engineering-history.npz"
    np.savez_compressed(destination, time_load_s=time, **mechanical,
                        q_dofs_microstrain=q, cable_layer_extension_mm=layers,
                        cable_mean_extension_mm=mean,
                        cable_depth_extension_difference_mm=difference,
                        valid_direction_region=np.isfinite(q).all(axis=-1))

    def probe(t):
        values = interpolate_time(raw_time, raw_values, np.asarray([t]))[0]
        observed = np.stack([
            interpolate_time(a["time_load_s"], a["target_from_known_load_reference_microstrain"].reshape(-1, 6), np.asarray([t]))[0].reshape(3, 2)
            for a in aligned
        ])
        # Reuse the event-based bracket, including missing optical support.
        shifts = np.asarray(alignment["offset_bracket_s"]) - alignment["offset_s"]
        at_bracket = np.stack([
            interpolate_time(a["time_load_s"], a["target_from_known_load_reference_microstrain"].reshape(-1, 6), t + shifts).reshape(2, 3, 2)
            for a in aligned
        ])
        layer, mean, depth = regional_extensions(observed, lengths)
        return {"time_load_s": t, **mechanical_outputs(values, reference, channels),
                "q_dofs_microstrain": observed,
                "q_at_clock_bracket_microstrain": at_bracket,
                "cable_layer_extension_mm": layer,
                "cable_mean_extension_mm": mean,
                "cable_depth_extension_difference_mm": depth}

    crossing_probes = {}
    for level in settings["engineering_alignment"]["force_probe_levels_kn"]:
        crossing_probes[f"{level:g}"] = {name: probe(t) for name, t in force_crossings(raw_time, force, level).items()}
    paired = crossing_probes[f"{settings['cl100h_alignment']['same_force_probe_kn']:g}"]
    contrast = paired["falling_after_peak"]["q_dofs_microstrain"] - paired["rising_before_peak"]["q_dofs_microstrain"]
    _, mean_contrast, depth_contrast = regional_extensions(contrast, lengths)
    return {
        "history_file": destination.relative_to(ROOT).as_posix(),
        "source_aligned_records": [row["file"] for row in alignment["aligned_records"]],
        "directions": ["1", "2"], "regions": ["left", "central", "right"],
        "q_components": ["q_m", "q_b"], "layer_order": ["bottom", "top"],
        "samples": len(time),
        "complete_all_directions_samples": int(np.isfinite(q).all(axis=(1, 2, 3)).sum()),
        "time_load_range_s": [time[0], time[-1]],
        "mechanical_reference_channels": channels,
        "mechanical_reference_values": reference,
        "optical_reference_targets_microstrain": np.stack([a["known_load_reference_target_microstrain"] for a in aligned]),
        "reference_window_load_s": settings["cl100h_alignment"]["known_load_reference_window_s"],
        "global_peak": probe(float(raw_time[force.argmax()])),
        "force_probes": crossing_probes,
        "same_force_history_change": {
            "force_kn": settings["cl100h_alignment"]["same_force_probe_kn"],
            "q_dofs_microstrain": contrast,
            "cable_mean_extension_mm": mean_contrast,
            "cable_depth_extension_difference_mm": depth_contrast,
            "whole_line_q_change_microstrain": np.einsum("r,drc->dc", lengths / lengths.sum(), contrast),
        },
        "reaction_balance": "For upward cone load P, upward trestle reaction T, downward rods R and constant weight W: P+T-R-W=0, hence delta_T=sum(delta_R)-delta_P. Recorded changes identify reaction redistribution, not absolute T or W.",
        "reference_meaning": "Fixed first known-load reference. Rod forces and LVDTs remain measured increments; no virgin-material state or zero preload is inferred.",
        "sampling": "Load channels are sampled on direction 1 DOFS times; direction 2 uses its corrected acquisition origin and adjacent-frame interpolation. Missing values remain missing. The saved series is an observation path, not displacement solution labels.",
    }


def optical_region_probes(lengths):
    reports = {}
    for specimen in ("CL100H", "BL100H"):
        for direction in (1, 2):
            path = PROCESSED / f"{specimen}-d{direction}-regional.npz"
            with np.load(path) as data:
                q = data["target_microstrain"].astype(float)
                complete = np.flatnonzero(data["common_mask"])
                selected = {
                    "largest_central_depth_difference": complete[np.argmax(abs(q[complete, 1, 1]))],
                    "largest_left_right_depth_difference": complete[np.argmax(abs(q[complete, 2, 1] - q[complete, 0, 1]))],
                }
                rows = {}
                for label, i in selected.items():
                    layer, mean, depth = regional_extensions(q[i], lengths)
                    signed = (lengths / lengths.sum()) * q[i, :, 1]
                    rows[label] = {
                        "time_pair_local_s": float(data["time_s"][i]), "q_dofs_microstrain": q[i],
                        "cable_layer_extension_mm": layer, "cable_mean_extension_mm": mean,
                        "cable_depth_extension_difference_mm": depth,
                        "whole_line_depth_extension_difference_mm": depth.sum(),
                        "opposite_sign_cancellation_fraction": 1 - abs(signed.sum()) / abs(signed).sum(),
                        "right_minus_left_depth_extension_difference_mm": depth[2] - depth[0],
                    }
                reports[f"{specimen}-d{direction}"] = rows
    return reports


def json_values(value):
    """Preserve missing observations as JSON null; no support is imputed."""
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    settings = json.loads((ROOT / "inputs/jansson_mechanics.json").read_text(encoding="utf-8"))
    alignment = json.loads((ROOT / "results/ideation/B2_mechanics_results.json").read_text(encoding="utf-8"))["cl100h_alignment"]
    regions = json.loads((ROOT / "inputs/regional_readout.json").read_text(encoding="utf-8"))["regions"]
    lengths = np.asarray([np.diff(row["interval_m"])[0] for row in regions])
    result = {
        "dataset_doi": settings["dataset_doi"], "region_lengths_m": lengths,
        "extension_definition": "l_bottom=L*(q_m-q_b)*1e-3 mm; l_top=L*(q_m+q_b)*1e-3 mm; l_mean=L*q_m*1e-3 mm; l_top-l_bottom=2*L*q_b*1e-3 mm, with q in microstrain and L in metres.",
        "transfer_identity": "For the unaveraged fibre field at each depth, l_fibre-l_host = slip_D(b)-slip_D(a), where slip_D=u_fibre-u_host. Host endpoint displacement includes internal crack openings. The measured l_D additionally includes instrument averaging and requires the corresponding spatial operator; no V9 or gauge transfer coefficient is fitted here.",
        "cl100h": cl_history(lengths, settings, alignment),
        "optical_regional_probes": optical_region_probes(lengths),
    }
    destination = ROOT / "results/ideation/B2_engineering_alignment.json"
    destination.write_text(json.dumps(json_values(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": destination.relative_to(ROOT).as_posix(),
                      "history": result["cl100h"]["history_file"],
                      "samples": result["cl100h"]["samples"]}))


if __name__ == "__main__":
    main()
