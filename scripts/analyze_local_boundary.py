"""Check force-moment admissibility and the azimuth of local observations.

Only the existing measured histories and saved mechanical states are consumed.
Reaction polygons establish necessary balance conditions, not identified
boundary parameters or displacement solutions. Rod channels are never assigned
to physical corners on the strength of a fit.
"""

import itertools
import json
import time

import numpy as np
from scipy.spatial import ConvexHull
import torch

from calibrate_support_response import affine_fit, loading_window, local_invariants, measured_crossing
from initial_state_model import InitialStatePhysics, material_densities, relative_edge_probes, surface_faces
from stateful_physics import ROOT


def convex_polygon(points):
    points = np.unique(np.asarray(points), axis=0)
    return points[ConvexHull(points).vertices]


def polygon_halfplanes(polygon):
    """Outward normals and offsets for a counterclockwise convex polygon."""
    edge = np.roll(polygon, -1, axis=0) - polygon
    normal = np.column_stack([edge[:, 1], -edge[:, 0]])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    return normal, -np.einsum("ij,ij->i", normal, polygon)


def clip_polygon(polygon, normals, bounds):
    """Intersect a convex polygon with n.x <= b, retaining its geometry."""
    result = np.asarray(polygon, dtype=float).reshape(-1, 2)
    for normal, bound in zip(normals, bounds):
        if not len(result):
            break
        previous = result[-1]
        previous_distance = previous @ normal - bound
        vertices = []
        for current in result:
            distance = current @ normal - bound
            if (distance <= 0) != (previous_distance <= 0):
                fraction = previous_distance / (previous_distance - distance)
                vertices.append(previous + fraction * (current - previous))
            if distance <= 0:
                vertices.append(current)
            previous, previous_distance = current, distance
        result = np.asarray(vertices).reshape(-1, 2)
    return result


def minimum_radius(polygon):
    """Distance from the origin to a nonempty convex polygon."""
    normal, offset = polygon_halfplanes(polygon)
    if np.all(offset <= 1e-12):
        return 0.
    edge = np.roll(polygon, -1, axis=0) - polygon
    norm2 = np.einsum("ij,ij->i", edge, edge)
    fraction = np.divide(-np.einsum("ij,ij->i", polygon, edge), norm2,
                         out=np.zeros(len(edge)), where=norm2 > 0).clip(0, 1)
    return float(np.linalg.norm(polygon + fraction[:, None] * edge, axis=1).min())


def balance_polygon(force_kn, rods_kn, centres_m, weight_kn, trestle_polygon, jack_polygon):
    """Return admissible jack centroids under vertical force/moment balance.

    For upward P and T and downward R and centred W, T=W+sum(R)-P
    and T*r_T=sum(R_i*r_i)-P*e. A nonnegative distributed support load
    has its centroid in the convex hull of its footprint. With T=0 the
    same halfplanes impose zero remaining moment.
    """
    reaction = float(weight_kn + np.sum(rods_kn) - force_kn)
    moment = np.asarray(rods_kn) @ np.asarray(centres_m)
    normal, offset = polygon_halfplanes(trestle_polygon)
    if reaction < 0:
        return reaction, moment, False, np.empty((0, 2))
    lhs = -force_kn * normal
    rhs = -normal @ moment - offset * reaction
    centred = bool(np.all(rhs >= -1e-12))
    return reaction, moment, centred, clip_polygon(jack_polygon, lhs, rhs)


def moment_correspondence(history, reference_rods_kn, weight_kn, centres, trestle, jack, levels,
                         whole_footprint):
    measured = [measured_crossing(history, level) for level in levels]
    mappings = []
    stage_polygons = [[] for _ in levels]
    centred_counts = np.zeros(len(levels), dtype=int)
    for permutation in itertools.permutations(range(4)):
        intersection = jack.copy()
        rows = []
        for j, (level, record) in enumerate(zip(levels, measured)):
            rods = record["rod_force_increment_kn"][list(permutation)] + reference_rods_kn
            reaction, moment, centred, polygon = balance_polygon(
                level, rods, centres, weight_kn, trestle, jack)
            centred_counts[j] += centred
            stage_polygons[j].append(polygon)
            if len(polygon):
                normal, offset = polygon_halfplanes(trestle)
                intersection = clip_polygon(intersection, -level * normal,
                                            -normal @ moment - offset * reaction)
            else:
                intersection = np.empty((0, 2))
            rows.append(dict(force_kn=level, trestle_force_kn=reaction,
                             rod_first_moment_kn_m=moment,
                             self_balanced_diagonal_contrast_kn=float(
                                 rods @ np.prod(np.sign(centres), axis=1) / 4),
                             centred_load_admissible=centred))
        mappings.append(dict(channels_at_model_corners=[i + 1 for i in permutation],
            stages=rows, fixed_eccentricity_polygon_m=intersection,
            minimum_fixed_eccentricity_m=minimum_radius(intersection) if len(intersection) else None))
    summaries = []
    for j, (level, record) in enumerate(zip(levels, measured)):
        feasible = [polygon for polygon in stage_polygons[j] if len(polygon)]
        broad_polygons, broad_centred = [], 0
        for permutation in itertools.permutations(range(4)):
            rods = record["rod_force_increment_kn"][list(permutation)] + reference_rods_kn
            _, _, centred, polygon = balance_polygon(level, rods, centres, weight_kn,
                                                     whole_footprint, jack)
            broad_centred += centred
            if len(polygon):
                broad_polygons.append(polygon)
        summaries.append(dict(force_kn=level,
            measured_rod_increment_kn=record["rod_force_increment_kn"],
            trestle_force_kn=mappings[0]["stages"][j]["trestle_force_kn"],
            centred_admissible_mappings=int(centred_counts[j]),
            eccentric_admissible_mappings=len(feasible),
            minimum_eccentricity_over_mappings_m=min(map(minimum_radius, feasible)) if feasible else None,
            whole_footprint_centred_admissible_mappings=broad_centred,
            whole_footprint_minimum_eccentricity_m=min(map(minimum_radius, broad_polygons)) if broad_polygons else None))
    return dict(rod_centres_m=centres, trestle_polygon_m=trestle, jack_polygon_m=jack,
                whole_footprint_polygon_m=whole_footprint,
                stages=summaries, mappings=mappings,
                fixed_eccentricity_admissible_mappings=sum(bool(len(row["fixed_eccentricity_polygon_m"])) for row in mappings))


def azimuth_correspondence(p, states, forces, history, azimuth_step):
    angles = np.arange(0., 120., azimuth_step)
    delta = states - states[1]
    probes = []
    for angle in angles:
        operator, _ = relative_edge_probes(p.mesh, p.inverse, angle + np.array([0., 120., 240.]))
        probes.append(delta @ operator.T * 1000)
    values = np.stack(probes, axis=1)
    invariants = local_invariants(values)
    rows = []
    for j, force in enumerate(forces):
        if j < 2:
            continue
        measured = measured_crossing(history, force / 1000)
        rows.append(dict(force_kn=force / 1000,
            measured_lvdt_mm=measured["cone_relative_lvdt_increment_mm"],
            measured_mean_amplitude_mm=local_invariants(measured["cone_relative_lvdt_increment_mm"]),
            model_mean_range_mm=[invariants[j, :, 0].min(), invariants[j, :, 0].max()],
            model_amplitude_range_mm=[invariants[j, :, 1].min(), invariants[j, :, 1].max()]))
    return angles, values, rows


def regional_correspondence(history, saved, levels, lengths):
    """Retain observed left/right differences at the existing regional scale."""
    rows = []
    for level in levels:
        record = measured_crossing(history, level)
        j = np.searchsorted(history["time_load_s"], record["time_load_s"]) - 1
        alpha = ((record["time_load_s"] - history["time_load_s"][j])
                 / (history["time_load_s"][j + 1] - history["time_load_s"][j]))
        q = (1 - alpha)*history["q_dofs_microstrain"][j] + alpha*history["q_dofs_microstrain"][j+1]
        # These selected stages have all regional observations. Fail explicitly
        # if a changed input no longer supports this comparison.
        if not np.isfinite(q).all():
            raise ValueError(f"Incomplete regional measurements at {level:g} kN")
        model = np.flatnonzero(saved["force_n"] == level * 1000)
        row = dict(force_kn=level, time_load_s=record["time_load_s"], observed_q_microstrain=q,
            observed_right_minus_left_q_microstrain=q[:, 2] - q[:, 0],
            observed_right_minus_left_mean_extension_mm=(q[:, 2, 0]*lengths[2]-q[:, 0, 0]*lengths[0])*1e-3,
            observed_right_minus_left_depth_extension_mm=2*(q[:, 2, 1]*lengths[2]-q[:, 0, 1]*lengths[0])*1e-3)
        if len(model):
            row["saved_model_host_q_microstrain"] = saved["host_increment_microstrain"][model[0]]
        rows.append(row)
    return dict(directions=["1", "2"], regions=["left", "central", "right"], components=["q_m", "q_b"],
                meaning="Use the existing nominal event-clock alignment and fixed reference. Optical cable and host outputs remain different observation objects; no installation transfer is fitted here.",
                stages=rows)


def json_values(value):
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    config_file = ROOT / "inputs/local_boundary_correspondence.json"
    config = json.loads(config_file.read_text())
    calibration = json.loads((ROOT / config["support_report"]).read_text())
    support_config = json.loads((ROOT / calibration["configuration"]).read_text())
    initial = json.loads((ROOT / support_config["initial_model_report"]).read_text())
    model_config = json.loads((ROOT / support_config["initial_model_configuration"]).read_text())
    mesh = dict(np.load(ROOT / initial["mesh_file"]))
    p = InitialStatePhysics(mesh, initial["resolved_settings"], model_config,
                            material_densities(model_config["weight"])[0])
    history = dict(np.load(ROOT / config["history_file"]))
    geometry = p.settings["geometry"]
    trestle = convex_polygon(mesh["xyz"][mesh["trestle_faces"].ravel(), :2])
    substrate_faces, _ = surface_faces(mesh, 1, geometry["bottom_levels_m"][-1])
    whole_footprint = convex_polygon(mesh["xyz"][substrate_faces.ravel(), :2])
    faces, _ = surface_faces(mesh, 2, geometry["bottom_levels_m"][-1])
    selected = np.linalg.norm(mesh["xyz"][faces[:, :3], :2].mean(axis=1), axis=1) < geometry["jack_patch_radius_m"]
    jack = convex_polygon(mesh["xyz"][faces[selected].ravel(), :2])
    saved = dict(np.load(ROOT / calibration["cases"][0]["arrays"]))
    result = dict(configuration=config_file.relative_to(ROOT).as_posix(),
        source_history=config["history_file"], source_states=[case["arrays"] for case in calibration["cases"]],
        source_basis=[
            "references/papers/jansson2024_dataset/source.pdf, PDF pages 7 and 10, Figs. 3 and 7: 1 m rod square, finite bearing stack and trestles; no numbered force-to-corner registration.",
            "references/papers/jansson2025_lining/source.pdf, PDF page 6, Fig. 4b and full.md section 2.3: underside LVDT installation and 120-degree spacing; no dimensioned attachment endpoints or azimuth map.",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11460454/ section 4.3: local relative-motion and separate jack/load channels."
        ],
        moment_assumption=config["moment_assumption"],
        moment_admissibility=moment_correspondence(history, saved["rod_force_n"][1] / 1000,
            float(saved["trestle_force_n"][0]) / 1000, np.asarray(geometry["bolt_centres_m"]),
            trestle, jack, config["force_levels_kn"], whole_footprint),
        regional_correspondence=regional_correspondence(history, saved, config["regional_levels_kn"],
            np.diff(p.settings["observations"]["regions_m"], axis=1).ravel()),
        rod_windows=[], azimuth_cases=[], continuation=config["continuation"])
    for window in config["rod_slope_windows_kn"]:
        mask = loading_window(history, window)
        coefficients, rms = affine_fit(history["force_increment_kn"][mask], history["rod_force_increment_kn"][mask])
        result["rod_windows"].append(dict(force_window_kn=window, samples=int(mask.sum()),
            rod_slopes=coefficients[1], total_slope=float(coefficients[1].sum()), residual_rms_kn=rms))
    destination = ROOT / "data/processed/local_boundary_correspondence"
    destination.mkdir(parents=True, exist_ok=True)
    for case in calibration["cases"]:
        saved = dict(np.load(ROOT / case["arrays"]))
        angles, values, rows = azimuth_correspondence(p, saved["displacement_m"], saved["force_n"],
                                                     history, config["azimuth_step_deg"])
        file = destination / f"azimuth_rods_{case['rod_stiffness_n_per_m'] / 1e6:g}MN.npz"
        np.savez_compressed(file, force_n=saved["force_n"], azimuth_deg=angles, local_relative_increment_mm=values)
        result["azimuth_cases"].append(dict(rod_stiffness_n_per_m=case["rod_stiffness_n_per_m"],
                                            arrays=file.relative_to(ROOT).as_posix(), stages=rows))
        print(json.dumps(dict(completed_azimuth_case=case["rod_stiffness_n_per_m"],
                              elapsed_s=time.perf_counter()-started)), flush=True)
    result["elapsed_s"] = time.perf_counter() - started
    output = ROOT / "results/ideation/B2_local_boundary_correspondence.json"
    output.write_text(json.dumps(json_values(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
