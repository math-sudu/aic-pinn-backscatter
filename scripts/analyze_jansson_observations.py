"""Reconstruct fixed-support dual-depth strain changes from the SND records.

No mechanical solver, PINN, unknown-parameter fit, or RF observation is involved.
Spatial integration uses the actual DOFS coordinates. Missing support remains
missing; means are never renormalized over surviving gauges. Load files are
inventoried separately because their computer was not synchronized to DOFS.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/jansson2024"
PROCESSED = ROOT / "data/processed/jansson2024"
RESULTS = ROOT / "results/ideation"


def read_dofs_header(path):
    metadata = {}
    tare = None
    with Path(path).open(encoding="utf-8-sig") as stream:
        for row, line in enumerate(stream):
            cells = line.rstrip("\r\n").split("\t")
            if cells[0] == "x-axis":
                x = np.asarray(cells[3:], dtype=float)
                if np.any(np.diff(x) <= 0):
                    raise ValueError(f"Non-increasing DOFS coordinates: {path}")
                return metadata, tare, x, row + 1
            if cells[0] == "Tare":
                tare = np.asarray(cells[3:], dtype=float)
            elif len(cells) == 2:
                metadata[cells[0].rstrip(":")] = cells[1]
    raise ValueError(f"No x-axis record: {path}")


def integration_weights(x, left, right):
    """Exact integral of the piecewise-linear DOFS profile on a fixed interval."""
    x = np.asarray(x)
    if not x[0] <= left < right <= x[-1]:
        raise ValueError("Integration interval outside measured coordinates")
    knots = np.r_[left, x[(x > left) & (x < right)], right]
    trapezoid = np.zeros(len(knots))
    trapezoid[:-1] += np.diff(knots) / 2
    trapezoid[1:] += np.diff(knots) / 2
    j = np.clip(np.searchsorted(x, knots, side="right") - 1, 0, len(x) - 2)
    fraction = (knots - x[j]) / (x[j + 1] - x[j])
    weights = np.zeros(len(x))
    np.add.at(weights, j, trapezoid * (1 - fraction))
    np.add.at(weights, j + 1, trapezoid * fraction)
    return weights / (right - left)


def apply_weights(values, weights):
    support = weights > 0
    return values[:, support] @ weights[support]


def interpolate_time(time, values, target):
    """Interpolate only adjacent acquired frames, preserving missing values."""
    time = np.asarray(time)
    values = np.asarray(values)
    if np.any(np.diff(time) <= 0):
        raise ValueError("DOFS timestamps must increase")
    j = np.clip(np.searchsorted(time, target, side="right") - 1, 0, len(time) - 2)
    dt = time[j + 1] - time[j]
    fraction = (target - time[j]) / dt
    out = values[j] + fraction[:, None] * (values[j + 1] - values[j])
    valid = (target >= time[0]) & (target <= time[-1]) & (dt < 1.5 * np.median(np.diff(time)))
    out[~valid] = np.nan
    # An acquired frame is usable even if the following frame was dropped.
    exact = target == time[j]
    out[exact] = values[j[exact]]
    last = target == time[-1]
    out[last] = values[-1]
    return out


def read_load(path):
    import pandas as pd

    with Path(path).open(encoding="cp1252") as stream:
        lines = [stream.readline().rstrip("\r\n") for _ in range(5)]
    names = lines[3].split("\t")
    units = lines[4].split("\t")
    frame = pd.read_csv(path, sep="\t", skiprows=5, names=names,
                        decimal=",", encoding="cp1252")
    if len(names) != len(units) or not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError(f"Invalid load record schema: {path}")
    return lines[2], dict(zip(names, units)), frame


def load_strain_profile(specimen, record, config, setting):
    """Load the fixed central metre, referenced to the recorded first 10 s."""
    import pandas as pd

    directory = RAW / specimen / "DistributedOpticalFiberSensors"
    path = directory / f"{specimen}-{record}.tsv"
    metadata, tare, cable_x, skip = read_dofs_header(path)
    if metadata["Units"] != "microstrain" or metadata["X-Axis Units"] != "m":
        raise ValueError(f"Unexpected DOFS units: {path}")
    markers = np.asarray(config["heat_markers_m"])
    center = (markers[4] + markers[5]) / 2
    x = cable_x - center
    left, right = setting["target_interval_m"]
    selected = np.flatnonzero((x >= left - 0.02) & (x <= right + 0.02))
    frame = pd.read_csv(path, sep="\t", skiprows=skip, header=None,
                        usecols=[0] + (selected + 3).tolist())
    time = pd.to_datetime(frame.iloc[:, 0]).astype("int64").to_numpy() / 1e9
    values = frame.iloc[:, 1:].to_numpy(float)
    x = x[selected]
    baseline_rows = time < time[0] + setting["baseline_duration_s"]
    baseline = values[baseline_rows].mean(axis=0)
    values -= baseline
    return time, x, values, {
        "source": path.relative_to(ROOT).as_posix(), "row_count": len(time),
        "gauge_count": len(cable_x), "axis_row_one_based": skip,
        "tare_row_present": tare is not None, "header_rate": metadata["Measurement Rate per Channel"],
        "measured_median_dt_s": float(np.median(np.diff(time))),
        "time_start": str(frame.iloc[0, 0]), "time_end": str(frame.iloc[-1, 0]),
        "duration_s": float(time[-1] - time[0]),
        "dropped_frame_gaps": int(np.sum(np.diff(time) > 1.5 * np.median(np.diff(time)))),
        "center_cable_m": float(center), "central_heat_span_m": float(markers[5] - markers[4]),
        "baseline_rows": int(baseline_rows.sum()),
    }


def process_record(specimen, record, config, setting):
    import pandas as pd

    time, x, values, info = load_strain_profile(specimen, record, config, setting)
    left, right = setting["target_interval_m"]
    outputs = [apply_weights(values, integration_weights(x, left, right))]
    count = setting["candidate_gauges_per_depth"]
    pitch = (right - left) / count
    gauge_length = setting["candidate_gauge_length_m"]
    even = left + pitch * (np.arange(count) + 0.5)
    positions = np.stack([even, even + pitch * setting["candidate_shift_in_pitch"]])
    for candidate in positions:
        for position in candidate:
            weights = integration_weights(x, position - gauge_length / 2, position + gauge_length / 2)
            outputs.append(apply_weights(values, weights))
    middle_left, middle_right = setting["central_diagnostic_interval_m"]
    middle = apply_weights(values, integration_weights(x, middle_left, middle_right))
    fraction = (middle_right - middle_left) / (right - left)
    outputs.extend([middle, (outputs[0] - fraction * middle) / (1 - fraction)])
    reduced = np.stack(outputs, axis=1)
    hp_path = RAW / specimen / "DistributedOpticalFiberSensors" / f"{specimen}-{record}-hp.tsv"
    _, _, hp_x, hp_skip = read_dofs_header(hp_path)
    markers = np.asarray(config["heat_markers_m"])
    hp_indices = [int(np.argmin(abs(hp_x - marker))) for marker in markers]
    hp = pd.read_csv(hp_path, sep="\t", skiprows=hp_skip, header=None,
                     usecols=[0] + [j + 3 for j in hp_indices])
    hp_values = hp.iloc[:, 1:].to_numpy(float)
    hp_changes = np.nanmax(hp_values, axis=0) - np.nanmedian(hp_values[:20], axis=0)
    info["heat_marker_changes_microstrain"] = hp_changes.tolist()
    info["full_target_support_rows"] = int(np.isfinite(reduced[:, 0]).sum())
    return time, reduced, positions, info


def response_statistics(time, target, candidates, mask):
    valid = mask & np.isfinite(target).all(axis=1) & np.isfinite(candidates).all(axis=(1, 2))
    if not valid.any():
        raise ValueError("No common fully supported target and candidate observations")
    q = target[valid]
    errors = candidates[valid] - q[:, None, :]
    return {
        "rows": int(valid.sum()), "first_time_s": float(time[valid][0]),
        "last_time_s": float(time[valid][-1]),
        "target_min_microstrain": q.min(axis=0).tolist(),
        "target_max_microstrain": q.max(axis=0).tolist(),
        "target_rms_microstrain": np.sqrt(np.mean(q ** 2, axis=0)).tolist(),
        "candidate_rmse_microstrain": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
        "candidate_p95_abs_error_microstrain": np.percentile(abs(errors), 95, axis=0).tolist(),
        "best_candidate_by_joint_rmse": ["even", "shifted"][int(np.argmin(np.mean(errors ** 2, axis=(0, 2))))]
    }


def main():
    setting = json.loads((ROOT / "inputs/jansson_observations.json").read_text(encoding="utf-8"))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    records, comparisons, loads = {}, {}, {}
    for specimen, spec in setting["specimens"].items():
        for direction in [1, 2]:
            pair = {}
            for depth in ["bot", "top"]:
                record = f"{direction}-{depth}"
                time, values, positions, info = process_record(specimen, record, spec["records"][record], setting)
                records[f"{specimen}-{record}"] = info
                pair[depth] = (time, values)
            bottom_time, bottom = pair["bot"]
            top_time, top = pair["top"]
            overlap = (bottom_time >= top_time[0]) & (bottom_time <= top_time[-1])
            absolute_time = bottom_time[overlap]
            bottom = bottom[overlap]
            top = interpolate_time(top_time, top, absolute_time)
            time = absolute_time - min(bottom_time[0], top_time[0])
            target = np.stack([(top[:, 0] + bottom[:, 0]) / 2,
                               (top[:, 0] - bottom[:, 0]) / 2], axis=1)
            count = setting["candidate_gauges_per_depth"]
            top_gauges = top[:, 1:1 + 2 * count].reshape(-1, 2, count)
            bottom_gauges = bottom[:, 1:1 + 2 * count].reshape(-1, 2, count)
            candidate_top = top_gauges.mean(axis=2)
            candidate_bottom = bottom_gauges.mean(axis=2)
            candidates = np.stack([(candidate_top + candidate_bottom) / 2,
                                   (candidate_top - candidate_bottom) / 2], axis=2)
            regions = np.stack([(top[:, -2:] + bottom[:, -2:]) / 2,
                                (top[:, -2:] - bottom[:, -2:]) / 2], axis=2)
            key = f"{specimen}-d{direction}"
            np.savez_compressed(PROCESSED / f"{key}.npz", time_s=time,
                                target_microstrain=target.astype(np.float32),
                                candidate_microstrain=candidates.astype(np.float32),
                                regional_target_microstrain=regions.astype(np.float32),
                                top_gauges_microstrain=top_gauges.astype(np.float32),
                                bottom_gauges_microstrain=bottom_gauges.astype(np.float32),
                                positions_m=positions)
            # Fixed first 10 s provide a reference. Beyond that, report both the
            # earliest uninterrupted complete interval and all complete records.
            complete = np.isfinite(target).all(axis=1) & np.isfinite(candidates).all(axis=(1, 2))
            missing = np.flatnonzero(~complete & (time >= setting["baseline_duration_s"]))
            initial_end = int(missing[0]) if len(missing) else len(time)
            initial = np.arange(len(time)) < initial_end
            comparisons[key] = {
                "pair_row_count": len(time), "complete_rows": int(complete.sum()),
                "top_minus_bottom_first_timestamp_s": float(top_time[0] - bottom_time[0]),
                "all_complete": response_statistics(time, target, candidates, np.ones(len(time), bool)),
                "initial_complete_interval": response_statistics(time, target, candidates, initial),
                "npz": f"data/processed/jansson2024/{key}.npz"
            }
            regional_complete = complete & np.isfinite(regions).all(axis=(1, 2))
            index = np.flatnonzero(regional_complete)[np.argmax(abs(regions[regional_complete, 0, 1]))]
            fraction = np.diff(setting["central_diagnostic_interval_m"])[0] / np.diff(setting["target_interval_m"])[0]
            regional_magnitude = fraction * abs(regions[index, 0, 1]) + (1 - fraction) * abs(regions[index, 1, 1])
            comparisons[key]["signed_bending_diagnostic"] = {
                "selection": "Largest absolute central q_b among common fully supported records; no failure or load label.",
                "time_s": float(time[index]), "full_q_microstrain": target[index].tolist(),
                "central_q_microstrain": regions[index, 0].tolist(),
                "flanks_q_microstrain": regions[index, 1].tolist(),
                "cancellation_fraction": float(1 - abs(target[index, 1]) / regional_magnitude)
            }
            print(key, json.dumps(comparisons[key], ensure_ascii=False), flush=True)
        for path in sorted((RAW / specimen / "LoadDisplacement").glob("*.dat")):
            acquisition, units, frame = read_load(path)
            loads[path.name] = {
                "source": path.relative_to(ROOT).as_posix(), "acquisition_header": acquisition,
                "units": units, "row_count": len(frame),
                "duration_s": float(frame["Time"].iloc[-1] - frame["Time"].iloc[0]),
                "median_dt_s": float(np.median(np.diff(frame["Time"]))),
                "ranges_in_native_units": {name: [float(frame[name].min()), float(frame[name].max())] for name in frame.columns},
                "synchronization": "Not aligned to the independent DOFS clock; no load fraction or failure label assigned."
            }
    result = {
        "dataset_doi": setting["dataset_doi"], "target_components": ["q_m", "q_b"],
        "candidate_order": ["even", "shifted"], "records": records,
        "regional_order": ["central_0.4m", "remaining_flanks_0.6m"],
        "comparisons": comparisons, "load_records": loads,
        "interpretation": "Measured strain increments and provisional finite-support observation operators. Candidate errors quantify quadrature on these records, not PINN error or validated hardware performance."
    }
    (RESULTS / "B2_observation_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
