"""Locate regional extension, interface motion and fracture on the saved path.

The optical pass runs separately from Torch because this host's Arrow runtime
cannot be initialized reliably after the mechanical runtime. Neither pass fits
displacements, crack positions or material parameters to the measured profiles.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from analyze_jansson_observations import integration_weights, interpolate_time


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data/processed/loading_history_coverage/regional_optical_profiles.npz"
EDGES = np.array([-.5, -.37, -.2, .2, .37, .5])
REPRESENTATIVES = ("rising_300kn", "global_peak", "falling_300kn")


def integrate_profiles(x, values, edges=EDGES):
    """Return extensions in mm, preserving every missing supported sample."""
    weights = [integration_weights(x, a, b) * (b-a) * 1e-3
               for a, b in zip(edges[:-1], edges[1:])]
    return np.stack([values[..., w > 0] @ w[w > 0] for w in weights], axis=-1)


def regional_components(extensions, edges=EDGES):
    """Five geometric strips -> original three mean/depth-difference strains."""
    lengths = np.array([edges[2]-edges[0], edges[3]-edges[2], edges[5]-edges[3]])
    layers = np.stack([extensions[..., :2].sum(-1), extensions[..., 2],
                       extensions[..., 3:].sum(-1)], axis=-1)
    layers = np.moveaxis(layers, -2, -1) / lengths[:, None] * 1e3
    return np.stack([layers.mean(-1), (layers[..., 1]-layers[..., 0])/2], axis=-1)


def optical_profiles():
    from analyze_jansson_observations import load_strain_profile

    settings = json.loads((ROOT / "inputs/jansson_observations.json").read_text())
    readout = json.loads((ROOT / "inputs/regional_readout.json").read_text())
    timing = json.loads((ROOT / "results/ideation/B2_mechanics_results.json").read_text())["cl100h_alignment"]
    boundary = np.load(ROOT / "data/processed/loading_history_coverage/branch_path.npz")
    stages = boundary["time_load_s"]
    arrays = dict(stage_labels=boundary["stage_labels"], strip_edges_m=EDGES)
    all_extensions, all_brackets, all_gauges = [], [], []
    for direction in (1, 2):
        pair = []
        for depth in ("bot", "top"):
            record = f"{direction}-{depth}"
            pair.append(load_strain_profile("CL100H", record,
                settings["specimens"]["CL100H"]["records"][record], settings)[:3])
        bt, _, _ = pair[0]
        tt, _, _ = pair[1]
        overlap = (bt >= tt[0]) & (bt <= tt[-1])
        absolute = bt[overlap]
        aligned = np.load(ROOT / f"data/processed/jansson2024/CL100H-d{direction}-load-aligned.npz")
        if len(absolute) != len(aligned["time_load_s"]):
            raise ValueError("Raw pair does not match the saved mechanical alignment")
        np.testing.assert_allclose(absolute-absolute[0],
            aligned["time_load_s"]-aligned["time_load_s"][0], atol=5e-7, rtol=0)
        layers, brackets, gauges = [], [], []
        for depth, (time, x, values) in enumerate(pair):
            values = values[overlap] if depth == 0 else interpolate_time(time, values, absolute)
            values = values-values[aligned["reference_mask"]].mean(axis=0)
            sampled = interpolate_time(aligned["time_load_s"], values, stages)
            # Stage zero denotes the window mean, not its midpoint frame.
            sampled[0] = 0.
            arrays[f"d{direction}_z{depth}_x_m"] = x
            arrays[f"d{direction}_z{depth}_microstrain"] = sampled
            layers.append(integrate_profiles(x, sampled))
            bracket = np.stack([integrate_profiles(x, interpolate_time(
                aligned["time_load_s"], values, stages+offset-timing["offset_s"]))
                for offset in timing["offset_bracket_s"]], axis=1)
            bracket[0] = 0.
            brackets.append(bracket)
            positions = np.concatenate([np.linspace(r["interval_m"][0], r["interval_m"][1],
                2*r["gauges_per_depth"]+1)[1::2] for r in readout["regions"]])
            length = readout["gauge_length_m"]
            weights = [integration_weights(x, c-length/2, c+length/2) for c in positions]
            gauges.append(np.stack([sampled[:, w > 0] @ w[w > 0] for w in weights], axis=-1))
        all_extensions.append(np.stack(layers, axis=1))
        all_brackets.append(np.stack(brackets, axis=2))
        all_gauges.append(np.stack(gauges, axis=1))
    arrays["extension_mm"] = np.stack(all_extensions, axis=1)
    arrays["clock_bracket_extension_mm"] = np.stack(all_brackets, axis=2)
    arrays["gauge_increment_microstrain"] = np.stack(all_gauges, axis=1)
    reconstructed = regional_components(arrays["extension_mm"])
    np.testing.assert_allclose(reconstructed, boundary["q_dofs_microstrain"],
                               rtol=1e-7, atol=1e-6, equal_nan=True)
    np.savez_compressed(PROFILE, **arrays)
    print("Optical strips reconstruct the saved regional history and missing support", flush=True)


def sample_displacement(physics, displacement, body, points):
    """P1/P2 one-sided displacement values; nodes need not coincide with probes."""
    from quadratic_elements import shape_values

    elements = np.flatnonzero(physics.mesh["body"] == body)
    inverse = physics.inverse[elements]
    nodal = displacement.reshape(*displacement.shape[:-1], -1, 3)
    result = []
    for point in points:
        bary = np.einsum("i,eij->ej", np.r_[1., point], inverse)
        covering = np.flatnonzero(np.min(bary, axis=1) >= -1e-9)
        if not len(covering):
            raise ValueError(f"Sample {point} is outside body {body}")
        row = covering[0]
        nodes = physics.mesh["tetra"][elements[row]]
        weights = bary[row] if len(nodes) == 4 else shape_values(bary[row])
        result.append(np.einsum("...ac,a->...c", nodal[..., nodes, :], weights))
    return np.stack(result, axis=-2)


def material_line_extensions(physics, displacement, *, samples=151):
    """Sample deformed material lines, subtracting the first field's length.

    Return mm with axes state, direction, region, bottom/top depth. This is a
    geometric observation of the supplied fields, not a finite-strain solve.
    """
    obs = physics.settings["observations"]
    result = np.empty((len(displacement), len(obs["directions"]),
                       len(obs["regions_m"]), len(obs["depths_m"])))
    for d, direction in enumerate(obs["directions"]):
        for r, (left, right) in enumerate(obs["regions_m"]):
            for z, depth in enumerate(obs["depths_m"]):
                points = np.zeros((samples, 3))
                points[:, direction] = np.linspace(left, right, samples)
                points[:, 2] = depth
                moved = points+sample_displacement(physics, displacement, 0, points)
                length = np.linalg.norm(np.diff(moved, axis=-2), axis=-1).sum(-1)
                result[:, d, r, z] = (length-length[0])*1e3
    return result


def finite_values(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, None).tolist()


def history_change(baseline, labels, extensions, arrays, saved, network):
    """Compare regional histories with their saved pre-update physical state."""
    import torch

    source = json.loads(baseline.read_text())
    field = np.load(ROOT/source["field_profiles"])
    np.testing.assert_array_equal(field["stage_labels"], labels)
    checkpoint = ROOT/source["checkpoint"]
    before = np.load(checkpoint.with_name(checkpoint.stem+"_query.npz"))
    np.testing.assert_array_equal(before["parameters"], arrays["parameters"])
    old_state = torch.load(checkpoint, weights_only=True)["network"]
    torch.testing.assert_close(network.reference_coefficients, old_state["reference_coefficients"],
                               rtol=0, atol=0)
    for key in before.files:
        # Physical reference observables are unchanged; residual/loss reductions
        # can have roundoff differences across the BLAS thread configurations.
        if key not in ("parameters", "control_residual_m", "mechanical_loss_by_step"):
            np.testing.assert_array_equal(arrays[key][:, 0], before[key][:, 0])
    nominal = np.flatnonzero(np.all(before["parameters"] == [1., -1.], axis=-1))[0]
    ntrain = len(saved["source_setup"]["training_parameters"])
    changes = {key: arrays[key]-before[key] for key in
               ("host_increment", "circuit_increment", "raw_gauge_strain", "local_relative_m", "jack_force_n")}
    transitions = []
    for first, last in (("rising_300kn", "falling_300kn"), ("global_peak", "falling_300kn"),
                        ("event_1_before", "event_1_after"), ("event_2_before", "event_2_after")):
        a, b = labels.index(first), labels.index(last)
        transitions.append(dict(first=first, last=last,
            before_strip_extension_mm=(field["strip_extension_mm"][b]-field["strip_extension_mm"][a]).tolist(),
            after_strip_extension_mm=(extensions[b]-extensions[a]).tolist(),
            host_change_microstrain=((changes["host_increment"][nominal, b]
                -changes["host_increment"][nominal, a])*1e6).tolist(),
            circuit_change_microstrain=((changes["circuit_increment"][nominal, b]
                -changes["circuit_increment"][nominal, a])*1e6).tolist()))
    return dict(baseline=baseline.relative_to(ROOT).as_posix(),
        fixed_reference_coefficients_and_physical_reference_observations="unchanged",
        material_ranges=[dict(group=name,
            maximum_host_change_microstrain=float(abs(changes["host_increment"][selection]).max()*1e6),
            maximum_circuit_change_microstrain=float(abs(changes["circuit_increment"][selection]).max()*1e6),
            maximum_gauge_change_microstrain=float(abs(changes["raw_gauge_strain"][selection]).max()*1e6))
            for name, selection in (("training", slice(0, ntrain)), ("held", slice(ntrain, None)))],
        nominal_history=[dict(stage=label,
            strip_extension_change_mm=(extensions[step]-field["strip_extension_mm"][step]).tolist(),
            host_change_microstrain=(changes["host_increment"][nominal, step]*1e6).tolist(),
            circuit_change_microstrain=(changes["circuit_increment"][nominal, step]*1e6).tolist(),
            local_mean_change_mm=float(changes["local_relative_m"][nominal, step].mean()*1e3),
            jack_force_change_kn=float(changes["jack_force_n"][nominal, step]/1e3))
            for step, label in enumerate(labels)],
        transitions=transitions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-only", action="store_true")
    parser.add_argument("--reuse-optical", action="store_true")
    parser.add_argument("--name", default="controlled_fracture_amplitude")
    parser.add_argument("--output", default="results/ideation/B2_regional_redistribution.json")
    parser.add_argument("--field-name", default="regional_redistribution")
    parser.add_argument("--baseline", help="Earlier regional report to compare without overwriting its fields")
    parser.add_argument("--depth-kinematics", choices=["radial", "flexure"], default="radial")
    args = parser.parse_args()
    if args.optical_only:
        optical_profiles()
        return
    if not args.reuse_optical:
        subprocess.run([sys.executable, __file__, "--optical-only"], check=True)

    import torch
    from controlled_fracture_history import ControlledFractureProblem
    from fixed_reference_history import OUTPUT
    from weighted_readout import FixedShuntReadout, uniform_parameters
    from regional_opening_basis import opening_fields
    from spatial_fracture import elastic_stress, stress_components

    torch.set_num_threads(2)
    record = OUTPUT / f"{args.name}.json"
    saved = json.loads(record.read_text())
    problem = ControlledFractureProblem(saved["configuration"])
    p = problem.physics
    network = problem.load_network(exact_control=True, fixed_reference=saved["fixed_reference"])
    network.load_state_dict(torch.load(record.with_suffix(".pt"), weights_only=True)["network"])
    optical = np.load(PROFILE)
    labels = list(problem.boundary["stage_labels"])
    np.testing.assert_array_equal(optical["stage_labels"], labels)
    with torch.no_grad():
        eta = p.tensor([[1., -1.]])
        displacement, force = network(eta)
        _, state = problem.equilibrium(displacement, force, eta)
        readout_settings = json.loads((ROOT / "inputs/regional_readout.json").read_text())
        readout = FixedShuntReadout(uniform_parameters(), readout_settings, p.tensor)
        host, circuit, raw = problem.base.responses(displacement, readout)
        strains = problem.fracture.strains(displacement)[0].numpy()
    previous = np.load(record.with_name(record.stem+"_query.npz"))
    nominal = np.flatnonzero(np.all(previous["parameters"] == [1., -1.], axis=-1))[0]
    for key, value in (("host_increment", host), ("circuit_increment", circuit),
                       ("raw_gauge_strain", raw), ("jack_force_n", force)):
        np.testing.assert_allclose(value[0].numpy(), previous[key][nominal], rtol=1e-9, atol=1e-13)
    u = displacement[0].numpy()
    du = u-u[:1]
    nodes = np.unique(np.round(np.r_[np.linspace(-.5, .5, 201), EDGES], 12))
    line_u, strip_extension, interface_line = [], [], []
    for direction in (0, 1):
        layer_u, extension = [], []
        for depth in p.settings["observations"]["depths_m"]:
            points = np.zeros((len(nodes), 3))
            points[:, direction], points[:, 2] = nodes, depth
            values = sample_displacement(p, du, 0, points)
            endpoints = values[:, np.searchsorted(nodes, EDGES), direction]
            extension.append(np.diff(endpoints, axis=-1)*1e3)
            layer_u.append(values)
        line_u.append(np.stack(layer_u, axis=1))
        strip_extension.append(np.stack(extension, axis=1))
        points = np.zeros((len(nodes), 3))
        points[:, direction] = nodes
        upper = sample_displacement(p, u, 0, points)
        lower = np.empty_like(upper)
        on_cone = abs(nodes) < p.settings["geometry"]["cone_top_radius_m"]
        for body, mask in ((1, ~on_cone), (2, on_cone)):
            lower[:, mask] = sample_displacement(p, u, body, points[mask])
        interface_line.append(upper-lower)
    extensions = np.stack(strip_extension, axis=1)
    reconstructed = regional_components(extensions)
    np.testing.assert_allclose(reconstructed, host[0].numpy()*1e6, rtol=1e-8, atol=1e-7)
    # No crack-opening values are inferred from the scalar constitutive history.
    points, volume = problem.fracture.points, problem.fracture.volumes
    radius = np.linalg.norm(points[:, :2], axis=-1)
    damage = state["bulk_damage"][0].numpy()
    displacement_line = np.stack(line_u, axis=1)
    gap_line = np.stack(interface_line, axis=1)
    # The current nonlocal radius sets a geometric trial-band width. It is not
    # fitted to V9 peaks and does not prescribe the final localization width.
    fields, mode_names = opening_fields(p.mesh, p.settings["geometry"]["cone_top_radius_m"],
        problem.config["nonlocal_radius_m"], p.settings["observations"]["depths_m"],
        depth_kinematics=args.depth_kinematics)
    fields *= 1e-6
    encoded = problem.coordinates.encode_nominal(fields)
    with torch.no_grad():
        decoded = (problem.coordinates.decode(p.tensor(encoded)[None])*p.scale)[0].numpy()
        np.testing.assert_allclose(decoded, fields, rtol=1e-7, atol=1e-13)
        mode_host = p.observations(p.tensor(fields))[0].numpy()*1e6
        mode_raw = p.tensor(fields) @ problem.base.raw_matrix.T
        mode_circuit = readout(mode_raw).numpy()*1e6
        mode_strain = problem.fracture.strains(p.tensor(fields)[None])[0]
        mode_stress = stress_components(elastic_stress(mode_strain,
            p.tensor(p.settings["bulk"]["top_young_pa"]), p.settings["bulk"]["poisson"]))
        mode_work = (.5*mode_strain*mode_stress*p.tensor(volume)[None, :, None]).numpy()
        total_work = mode_work.sum(axis=(1, 2))
        beyond_band = radius > p.settings["geometry"]["cone_top_radius_m"]+problem.config["nonlocal_radius_m"]
    mode_rank = int(np.linalg.matrix_rank(mode_host.reshape(len(fields), -1)))
    if mode_rank != len(fields):
        raise AssertionError("Geometric directions do not separate the retained regional outputs")
    rows = []
    for stage in REPRESENTATIVES:
        step = labels.index(stage)
        regional_fields = []
        for inner, outer in ((0., .2), (.2, .37), (.37, .5), (.5, 1.3)):
            mask = (radius >= inner) & (radius < outer)
            weight = volume[mask]
            regional_fields.append(dict(radial_interval_m=[inner, outer],
                volume_m3=float(weight.sum()),
                volume_mean_damage=float(damage[step, mask] @ weight / weight.sum()),
                maximum_damage=float(damage[step, mask].max()),
                damaged_volume_fraction=float(weight[damage[step, mask] > 0].sum()/weight.sum())))
        layers = optical["extension_mm"][step]
        probe = optical["gauge_increment_microstrain"][step]
        peaks = []
        for direction in (1, 2):
            depth_peaks = []
            for depth in (0, 1):
                x = optical[f"d{direction}_z{depth}_x_m"]
                profile = optical[f"d{direction}_z{depth}_microstrain"][step]
                side_peaks = []
                for left, right in ((-.5, -.2), (.2, .5)):
                    selected = np.flatnonzero((x >= left) & (x <= right))
                    maximum = selected[np.nanargmax(profile[selected])]
                    side_peaks.append(dict(coordinate_m=float(x[maximum]),
                                           strain_microstrain=float(profile[maximum])))
                depth_peaks.append(side_peaks)
            peaks.append(depth_peaks)
        rows.append(dict(stage=stage,
            host_strip_extension_mm=extensions[step].tolist(),
            optical_strip_extension_mm=finite_values(layers),
            optical_clock_bracket_strip_extension_mm=finite_values(optical["clock_bracket_extension_mm"][step]),
            optical_assumed_gauge_maximum_abs_increment_microstrain=np.nanmax(abs(probe), axis=-1).tolist(),
            optical_assumed_gauges_above_nominal_limit=np.sum(abs(probe) >
                readout_settings["sensor_nominal_strain_limit_microstrain"], axis=-1).tolist(),
            optical_side_peaks=peaks,
            interface_line_by_region=[[
                dict(maximum_normal_gap_mm=float(gap_line[step, direction, (nodes >= left) & (nodes <= right), 2].max()*1e3),
                     maximum_tangential_slip_mm=float(np.linalg.norm(gap_line[step, direction,
                         (nodes >= left) & (nodes <= right), :2], axis=-1).max()*1e3))
                for left, right in ((-.5, -.2), (-.2, .2), (.2, .5))] for direction in (0, 1)],
            host_microstrain=reconstructed[step].tolist(),
            fixed_circuit_microstrain=(circuit[0, step].numpy()*1e6).tolist(),
            radial_fracture=regional_fields))
    a, b = (labels.index(s) for s in ("rising_300kn", "falling_300kn"))
    optical_change = optical["extension_mm"][b]-optical["extension_mm"][a]
    host_change = extensions[b]-extensions[a]
    output = ROOT / args.output
    report = dict(checkpoint=record.with_suffix(".pt").relative_to(ROOT).as_posix(),
        nominal_parameters=[1., -1.], optical_profiles=PROFILE.relative_to(ROOT).as_posix(),
        field_profiles=(OUTPUT / f"{args.field_name}.npz").relative_to(ROOT).as_posix(),
        strip_edges_m=EDGES.tolist(),
        strip_definition="Fixed target boundaries +/-0.2 and +/-0.5 m, split at the physical cone-top radius +/-0.37 m",
        array_axes="extensions: direction, depth (bottom/top), strip; clock brackets prepend lower/upper offset",
        reference="The existing loaded reference; optical profiles preserve the original pair clock and missing support",
        representative_history=rows,
        branch_change=dict(first_stage=labels[a], last_stage=labels[b],
            host_strip_extension_mm=host_change.tolist(), optical_strip_extension_mm=finite_values(optical_change),
            optical_outside_cone_fraction_of_side_extension_change=np.stack([
                optical_change[..., 0]/optical_change[..., :2].sum(-1),
                optical_change[..., 4]/optical_change[..., 3:].sum(-1)], axis=-1).tolist()),
        geometric_trial_modes=dict(names=mode_names, amplitude_m=1e-6, depth_kinematics=args.depth_kinematics,
            half_width_m=problem.config["nonlocal_radius_m"],
            host_microstrain=mode_host.tolist(), fixed_circuit_microstrain=mode_circuit.tolist(),
            regional_rank=mode_rank,
            elastic_work=dict(young_pa=p.settings["bulk"]["top_young_pa"], poisson=p.settings["bulk"]["poisson"],
                normal_j=mode_work[..., :3].sum(axis=(1, 2)).tolist(),
                in_plane_shear_j=mode_work[..., 3].sum(axis=1).tolist(),
                transverse_shear_j=mode_work[..., 4:].sum(axis=(1, 2)).tolist(),
                transverse_shear_fraction=(mode_work[..., 4:].sum(axis=(1, 2))/total_work).tolist(),
                outside_band_transverse_shear_fraction_of_total=(mode_work[:, beyond_band, 4:].sum(axis=(1, 2))/total_work).tolist(),
                role="Undamaged elastic work of each unit kinematic trial, separating in-plane extension from transverse shear; not energy dissipated along the trained path"),
            role="Kinematic trial fields represented exactly by the existing neural coordinates. They separate side extension and depth contrast; optical amplitudes are not fitted."),
        field_interpretation="Displacement, interface jump and volume damage are evaluated at the saved neural prediction. Equivalent constitutive opening is not a geometric crack map.",
        gauge_interpretation="The 30 mm uniform kernel is applied to measured V9 strain increments as a support/range diagnostic. It is not a calibrated strain transfer for a separate installed gauge.",
        checks=["Raw optical strips reassemble the saved regional observations including missing support",
                "Nominal host, gauge, circuit and force outputs reconstruct the saved checkpoint",
                "One-sided P2 endpoint differences reassemble the existing integrated host targets",
                "Eight geometric trial fields reconstruct in neural coordinates and independently change regional outputs"])
    if args.baseline:
        report["history_change"] = history_change(ROOT/args.baseline, labels, extensions, previous, saved, network)
    np.savez_compressed(ROOT/report["field_profiles"], stage_labels=np.asarray(labels),
        line_coordinate_m=nodes, displacement_increment_m=displacement_line,
        interface_jump_m=gap_line, strip_extension_mm=extensions,
        bulk_points_m=points, bulk_quadrature_volume_m3=volume,
        bulk_damage=damage, bulk_strain=strains,
        trial_mode_names=np.asarray(mode_names), trial_mode_displacement_m=fields)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=output.relative_to(ROOT).as_posix(), branch_change=report["branch_change"])), flush=True)


if __name__ == "__main__":
    main()
