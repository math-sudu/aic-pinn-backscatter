"""Recover material-test channels and an event-based CL100H load correspondence.

The first mechanical/DOFS event estimates a clock offset; the second is a check.
No load-strain constitutive relation is fitted. Missing DOFS support is retained,
and loads are never extrapolated before the load record or over a recording gap.
"""

import json
from datetime import datetime

import numpy as np
from analyze_jansson_observations import ROOT, RAW, PROCESSED, read_load, interpolate_time
from jansson_materials import read_cube


def material_summary():
    rows, arrays = {}, {}
    for path in sorted((RAW / "MaterialData").rglob("*")):
        if not path.is_file():
            continue
        key = path.stem.replace("-", "_")
        row = {"source": path.relative_to(ROOT).as_posix()}
        if path.suffix.lower() == ".tra":
            metadata, columns, units, values, area = read_cube(path)
            strength = float(values[:, 1].max() * 1000 / area)
            row.update(kind="cube_compression", metadata=metadata,
                       loaded_area_mm2=area, compressive_strength_mpa=strength,
                       declared_strength_mpa=float(metadata["Strength"][0].replace(",", ".")))
            row["calculated_minus_declared_strength_mpa"] = strength - row["declared_strength_mpa"]
        elif path.suffix.lower() == ".dat":
            header, unit_map, frame = read_load(path)
            columns, units, values = list(frame), list(unit_map.values()), frame.to_numpy()
            row.update(kind="wedge_splitting" if "Split_force" in columns else "core_tension",
                       acquisition_header=header,
                       channel_ranges={name: [float(frame[name].min()), float(frame[name].max())]
                                       for name in columns})
            if "Split_force" in columns:
                index = int(frame["Split_force"].argmin())
                row["peak_compressive_split_channel_kn"] = float(-frame.iloc[index]["Split_force"])
                row["cmod_at_split_channel_peak_mm"] = float(frame.iloc[index]["Ch 1 CMOD"])
            else:
                row["peak_tensile_force_kn"] = float(frame["Ch 1 Force"].max())
        else:
            raise ValueError(f"Unrecognized material data file: {path}")
        row.update(columns=columns, units=units, samples=len(values), array_key=key)
        rows[key], arrays[key] = row, values
    if not rows:
        raise FileNotFoundError("Fetch MaterialData before running this analysis")
    np.savez_compressed(PROCESSED / "material-tests.npz", **arrays)
    groups = {}
    for group, prefix in [("substrate_cast1", "cast1"), ("substrate_cast2", "cast2"), ("top_layer", "cast3")]:
        values = [row["compressive_strength_mpa"] for key, row in rows.items() if key.startswith(prefix)]
        groups[group] = {"n": len(values), "mean_mpa": float(np.mean(values)),
                         "range_mpa": [float(min(values)), float(max(values))]}
    return {"records": rows, "cube_groups": groups}


def displacement_event_interval(time, displacement, window):
    """Bracket the largest observed displacement increment in a named event."""
    index = np.flatnonzero((time[:-1] >= window[0]) & (time[1:] <= window[1]))
    rate = np.diff(displacement) / np.diff(time)
    peak = index[np.argmax(rate[index])]
    return [float(time[peak]), float(time[peak + 1])]


def first_missing_event_interval(time, values, window):
    """Keep the complete frames bracketing an event's first loss of support."""
    valid = np.isfinite(values.reshape(len(time), -1)).all(axis=1)
    index = np.flatnonzero((time >= window[0]) & (time <= window[1]) & ~valid)
    if not len(index):
        raise ValueError(f"No DOFS loss of support in event window {window}")
    first = index[0]
    before = np.flatnonzero(valid[:first])
    after = np.flatnonzero(valid[first + 1:]) + first + 1
    if not len(before) or not len(after):
        raise ValueError("Cannot bracket missing event at record boundary")
    return [float(time[before[-1]]), float(time[after[0]])]


def offset_interval(load_interval, dofs_interval):
    return [dofs_interval[0] - load_interval[1], dofs_interval[1] - load_interval[0]]


def regional_clock_offsets(records, specimen, files):
    """Map pair-local times to one origin on the shared DOFS acquisition clock."""
    starts = {}
    for name in files:
        direction = name.split("-d", 1)[1].split("-", 1)[0]
        starts[name] = min(datetime.fromisoformat(records[f"{specimen}-{direction}-{depth}"]["time_start"])
                           for depth in ("bot", "top"))
    origin = min(starts.values())
    return origin.isoformat(), {name: (start - origin).total_seconds() for name, start in starts.items()}


def same_force_probe(load, dofs, offset, offset_bracket, setting):
    force = -load["Ch 1 Force"].to_numpy()
    time = load["Time"].to_numpy()
    level = setting["same_force_probe_kn"]
    peak = int(force.argmax())
    rising = np.flatnonzero((force[:-1] < level) & (force[1:] >= level) & (np.arange(len(force) - 1) < peak))
    falling = np.flatnonzero((force[:-1] > level) & (force[1:] <= level) & (np.arange(len(force) - 1) >= peak))
    if not len(rising) or not len(falling):
        raise ValueError("The requested force is not crossed on both branches")
    index = np.array([rising[0], falling[0]])
    times = time[index] + ((level - force[index]) / (force[index + 1] - force[index])) * (time[index + 1] - time[index])
    values, left_endpoint_values = {}, {}
    for name, a in dofs.items():
        target = interpolate_time(a["time_s"], a["target_microstrain"].reshape(-1, 6), times + offset)
        if not np.isfinite(target).all():
            raise ValueError("Incomplete DOFS support at the same-force probe")
        values[name] = target.reshape(2, 3, 2).tolist()
        # Check the named left-region history contrast at both offset endpoints.
        endpoint_times = (times[:, None] + np.array(offset_bracket)).ravel()
        left = interpolate_time(a["time_s"], a["target_microstrain"][:, 0], endpoint_times)
        if not np.isfinite(left).all():
            raise ValueError("Incomplete left-region support at alignment endpoints")
        left_endpoint_values[name] = left.reshape(2, 2, 2).tolist()
    return {"compression_force_kn": level, "branch_order": ["rising_before_global_peak", "falling_after_global_peak"],
            "load_times_s": times.tolist(), "dofs_times_s": (times + offset).tolist(),
            "regional_targets_microstrain": values,
            "left_region_at_offset_endpoints_microstrain": left_endpoint_values}


def align_cl100h(setting):
    _, units, load = read_load(ROOT / setting["load_file"])
    load_time = load["Time"].to_numpy()
    mean_lvdt = load[setting["displacement_channels"]].mean(axis=1).to_numpy()
    dofs = {name: dict(np.load(PROCESSED / name)) for name in setting["dofs_files"]}
    observation_report = json.loads((ROOT / "results/ideation/B2_observation_results.json").read_text(encoding="utf-8"))
    origin, clock_offsets = regional_clock_offsets(observation_report["records"], "CL100H", dofs)
    for name, a in dofs.items():
        a["time_s"] = a["time_s"] + clock_offsets[name]
    events = []
    for event in setting["events"]:
        load_interval = displacement_event_interval(load_time, mean_lvdt, event["load_window_s"])
        intervals = [first_missing_event_interval(a["time_s"], a["target_microstrain"],
                                                 event["dofs_window_s"]) for a in dofs.values()]
        # Union retains the acquisition/support uncertainty of both directions.
        dofs_interval = [min(x[0] for x in intervals), max(x[1] for x in intervals)]
        events.append({"role": event["role"], "load_interval_s": load_interval,
                       "dofs_interval_s": dofs_interval,
                       "compatible_offset_interval_s": offset_interval(load_interval, dofs_interval)})
    estimation = events[0]["compatible_offset_interval_s"]
    offset = float(np.mean(estimation))
    check = events[1]["compatible_offset_interval_s"]
    aligned = []
    channel_names = [c for c in load if c != "Time"]
    channel_values = load[channel_names].to_numpy()
    for name, a in dofs.items():
        time = a["time_s"]
        query = time - offset
        known_load = interpolate_time(load_time, channel_values, query)
        # The event bracket is propagated to load uncertainty without extrapolation.
        endpoints = np.stack([interpolate_time(load_time, channel_values, time - delta)
                              for delta in estimation], axis=1)
        known = np.isfinite(known_load).all(axis=1)
        reference = ((query >= setting["known_load_reference_window_s"][0]) &
                     (query < setting["known_load_reference_window_s"][1]) & known &
                     np.isfinite(a["target_microstrain"]).all(axis=(1, 2)))
        if not reference.any():
            raise ValueError("No complete samples in the known-load reference window")
        reference_target = a["target_microstrain"][reference].mean(axis=0)
        output = name.replace("-regional", "-load-aligned")
        np.savez_compressed(PROCESSED / output, time_dofs_s=time, time_load_s=query,
                            load_values=known_load, load_at_offset_endpoints=endpoints,
                            target_microstrain=a["target_microstrain"],
                            target_from_known_load_reference_microstrain=a["target_microstrain"] - reference_target,
                            known_load_reference_target_microstrain=reference_target,
                            known_load_reference_values=known_load[reference].mean(axis=0),
                            known_load_mask=known, reference_mask=reference)
        force_index = channel_names.index("Ch 1 Force")
        smooth = (known & (time - estimation[0] < events[0]["load_interval_s"][0]) &
                  np.isfinite(endpoints).all(axis=(1, 2)))
        endpoint_change = np.abs(endpoints[smooth, 0, force_index] - endpoints[smooth, 1, force_index])
        aligned.append({"file": (PROCESSED / output).relative_to(ROOT).as_posix(),
                        "source_dofs": name, "channels": channel_names,
                        "pair_origin_after_shared_origin_s": clock_offsets[name],
                        "units": [units[c] for c in channel_names],
                        "known_load_samples": int(known.sum()), "reference_samples": int(reference.sum()),
                        "original_baseline_precedes_load_record": bool(query[0] + 10 < load_time[0]),
                        "reference_target_microstrain": reference_target.tolist(),
                        "before_event_for_entire_clock_bracket_force_spread_kn": {
                            "median": float(np.median(endpoint_change)), "max": float(endpoint_change.max())}})
    return {"shared_dofs_origin": origin, "offset_s": offset, "offset_bracket_s": estimation, "events": events,
            "second_event_contains_estimated_offset": bool(check[0] <= offset <= check[1]),
            "event_midpoint_offset_difference_s": float(np.mean(check) - offset),
            "same_force_probe": same_force_probe(load, dofs, offset, estimation, setting),
            "aligned_records": aligned,
            "scope": "CL100H only, constant offset inferred from event correspondence. No BL100H mapping or elastic-stage label is inferred."}


def main():
    setting = json.loads((ROOT / "inputs/jansson_mechanics.json").read_text(encoding="utf-8"))
    result = {"dataset_doi": setting["dataset_doi"], "materials": material_summary(),
              "cl100h_alignment": align_cl100h(setting["cl100h_alignment"])}
    path = ROOT / "results/ideation/B2_mechanics_results.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"cube_groups": result["materials"]["cube_groups"],
                      "offset_s": result["cl100h_alignment"]["offset_s"],
                      "offset_bracket_s": result["cl100h_alignment"]["offset_bracket_s"],
                      "second_event_contains_offset": result["cl100h_alignment"]["second_event_contains_estimated_offset"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
