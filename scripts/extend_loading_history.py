"""Extend the shared PINN path to the measured peak and second 300-kN branch.

The saved network is an initialization, not a trained high-load solution. Force
and support admissibility, measured path order and history derivatives determine
what physics must be resolved before using this path for sensor synthesis.
"""

import copy
import json
import time

import numpy as np
import torch

from analyze_engineering_alignment import force_crossings, json_values, mechanical_outputs
from analyze_jansson_observations import interpolate_time
from analyze_local_boundary import polygon_halfplanes
from analyze_regional_material import stress_summary
from boundary_history import support_pressure_load, support_quadrature
from fixed_reference_history import ROOT, OUTPUT, FixedReferenceNetwork, MaterialHistoryProblem
from weighted_readout import FixedShuntReadout, uniform_parameters


def extension_map(old_force, new_force):
    """Preserve the old prefix; use its last secant only as a starting guess."""
    old_force, new_force = np.asarray(old_force), np.asarray(new_force)
    if len(new_force) <= len(old_force) or not np.array_equal(new_force[:len(old_force)], old_force):
        raise ValueError("The old physical path must be an exact prefix")
    interval = old_force[-1]-old_force[-2]
    if interval == 0:
        raise ValueError("The last two source forces cannot define a secant")
    result = np.zeros((len(new_force), len(old_force)))
    result[:len(old_force)] = np.eye(len(old_force))
    alpha = (new_force[len(old_force):]-old_force[-2])/interval
    result[len(old_force):, -2] = 1-alpha
    result[len(old_force):, -1] = alpha
    return result


def native_load(path, channels):
    """Read the DAT numerical table without loading the Arrow/Pandas runtime."""
    with path.open(encoding="cp1252") as stream:
        header = [stream.readline().rstrip("\r\n") for _ in range(5)]
        values = np.loadtxt((line.replace(",", ".") for line in stream), delimiter="\t")
    names = header[3].split("\t")
    units = dict(zip(names, header[4].split("\t")))
    if units["Time"] != "s" or units["Ch 1 Force"] != "kN":
        raise ValueError("Unexpected mechanical time/force units")
    return values[:, names.index("Time")], values[:, [names.index(name) for name in channels]]


def event_path(time_s, force_kn, config, events):
    """Choose native force crossings and jump endpoints, never sort by force."""
    end = force_crossings(time_s, force_kn, config["terminal_falling_force_kn"])["falling_after_peak"]
    knots = [(force_crossings(time_s, force_kn, level)["rising_before_peak"], f"rising_{level:g}kn")
             for level in config["additional_rising_force_kn"]]
    knots.append((float(time_s[np.argmax(force_kn)]), "global_peak"))
    for i, event in enumerate(events, start=1):
        knots.extend((float(t), f"event_{i}_{side}") for t, side in
                     zip(event["load_interval_s"], ["before", "after"]))
    for first, second in zip(events[:-1], events[1:]):
        selected = np.flatnonzero((time_s > first["load_interval_s"][1])
                                  & (time_s < second["load_interval_s"][0]))
        for name, pick in [("minimum", np.argmin), ("maximum", np.argmax)]:
            j = selected[pick(force_kn[selected])]
            knots.append((float(time_s[j]), f"between_events_{name}"))
    knots.append((end, f"falling_{config['terminal_falling_force_kn']:g}kn"))
    knots.sort()
    times = np.asarray([row[0] for row in knots])
    if np.any(np.diff(times) <= 0) or times[-1] != end:
        raise ValueError("Event path has duplicate or out-of-scope knots")
    return times, np.asarray([row[1] for row in knots])


def admissibility(force, rods, weight, centres, eccentricity, polygon):
    """Nonnegative support and centroid feasibility in kN and metres."""
    reaction = weight+rods.sum(axis=1)-force
    moment = rods@centres-force[:, None]*eccentricity
    normal, offset = polygon_halfplanes(polygon)
    margin = (-moment@normal.T-reaction[:, None]*offset).min(axis=1)
    return reaction, margin


def material_curve_evidence():
    """Describe the named force channel without inferring its wedge conversion."""
    records = json.loads((ROOT/"results/ideation/B2_mechanics_results.json").read_text())["materials"]["records"]
    arrays = np.load(ROOT/"data/processed/jansson2024/material-tests.npz")
    rows = []
    for name, record in records.items():
        if record["kind"] != "wedge_splitting":
            continue
        value = arrays[name]
        cmod = value[:, record["columns"].index("Ch 1 CMOD")]
        force = -value[:, record["columns"].index("Split_force")]
        peak = int(force.argmax())
        residual = []
        for opening in [0.5, 1., 2., 4.]:
            crossings = np.flatnonzero((cmod[:-1] < opening) & (cmod[1:] >= opening)
                                       & (np.arange(len(cmod)-1) >= peak))
            j = crossings[0]
            a = (opening-cmod[j])/(cmod[j+1]-cmod[j])
            residual.append(float(((1-a)*force[j]+a*force[j+1])/force[peak]))
        rows.append(dict(source=record["source"], force_channel="Split_force",
                         peak_named_channel_kn=float(force[peak]),
                         cmod_at_peak_mm=float(cmod[peak]), residual_named_channel_fraction=residual))
    return dict(cmod_probes_mm=[.5, 1., 2., 4.], records=rows,
        interpretation="The named Split_force channel retains substantial post-peak load. Its near-unity relation to Ch 1 Force does not identify a horizontal wedge-force conversion. CMOD is not local crack opening; the specimen/fixture geometry is required to infer a tensile law or fracture energy. See B2_fracture_path.json.")


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    config = json.loads((ROOT/"inputs/loading_history_coverage.json").read_text())
    source_path = OUTPUT/(config["source_network"]+".json")
    source = json.loads(source_path.read_text())
    problem = MaterialHistoryProblem(source["setup"])
    p = problem.physics
    network = FixedReferenceNetwork(problem.coordinates, problem.initial,
                                   source["setup"]["width"], source["setup"]["path_coordinates"])
    network.load_state_dict(torch.load(source_path.with_suffix(".pt"), weights_only=True)["network"])
    mechanics = json.loads((ROOT/"inputs/jansson_mechanics.json").read_text())
    alignment = json.loads((ROOT/"results/ideation/B2_engineering_alignment.json").read_text())["cl100h"]
    timing = json.loads((ROOT/"results/ideation/B2_mechanics_results.json").read_text())["cl100h_alignment"]
    channels = alignment["mechanical_reference_channels"]
    time_s, values = native_load(ROOT/mechanics["cl100h_alignment"]["load_file"], channels)
    raw = mechanical_outputs(values, np.asarray(alignment["mechanical_reference_values"]), channels)
    times, labels = event_path(time_s, raw["force_kn"], config, timing["events"])
    sampled = interpolate_time(time_s, values, times)
    if not np.isfinite(sampled).all():
        raise ValueError("A selected mechanical knot crosses missing acquisition support")
    observed = mechanical_outputs(sampled, np.asarray(alignment["mechanical_reference_values"]), channels)
    boundary_config = json.loads((ROOT/"inputs/boundary_history.json").read_text())
    asym = json.loads((ROOT/boundary_config["asymmetric_report"]).read_text())
    old_report = json.loads((ROOT/"results/ideation/B2_boundary_history.json").read_text())
    mapping = np.asarray(asym["channels_at_model_corners"])-1
    old = problem.boundary
    forces = np.r_[old["force_n"], observed["force_kn"]*1000]
    rods = np.vstack([old["rod_force_n"], old["rod_force_n"][0]+observed["rod_force_increment_kn"][:, mapping]*1000])
    points, area = support_quadrature(p)
    centres = p.anchor_operator.numpy()[:, 2::3]@p.mesh["xyz"][:, :2]
    gravity_moment = p.gravity.numpy()[2::3]@p.mesh["xyz"][:, :2]
    eccentricity = np.asarray(asym["eccentricity_m"])
    support, quadrature = list(old["support_load_n"]), list(old["support_quadrature_force_n"])
    for force, rod in zip(forces[len(old["force_n"]):], rods[len(old["force_n"]):]):
        total = -p.gravity.numpy().sum()+rod.sum()-force
        moment = rod@centres-force*eccentricity-gravity_moment
        vector, weights, _ = support_pressure_load(p, total, moment, area)
        support.append(vector)
        quadrature.append(weights)
    support, quadrature = np.asarray(support), np.asarray(quadrature)
    source_steps = len(old["force_n"])
    matrix = extension_map(old["force_n"], forces)
    labels = np.r_[np.asarray(["fixed_reference"]+[f"rising_{f/1000:g}kn" for f in old["force_n"][1:]]), labels]
    prefix_times = [np.mean(alignment["reference_window_load_s"])] + [r["time_load_s"] for r in old_report["cases"][0]["stages"]]
    all_times = np.r_[prefix_times, times]
    q = []
    for direction in (1, 2):
        optical = np.load(ROOT/f"data/processed/jansson2024/CL100H-d{direction}-load-aligned.npz")
        q.append(interpolate_time(optical["time_load_s"],
            optical["target_from_known_load_reference_microstrain"].reshape(-1, 6), all_times).reshape(-1, 3, 2))
    q = np.stack(q, axis=1)
    q[0] = 0.
    measured_all = mechanical_outputs(interpolate_time(time_s, values, all_times),
                                    np.asarray(alignment["mechanical_reference_values"]), channels)
    for key in measured_all:
        measured_all[key][0] = 0.
    measured_all["force_kn"][0] = old["force_n"][0]/1000
    destination = ROOT/"data/processed/loading_history_coverage"
    destination.mkdir(parents=True, exist_ok=True)
    file = destination/"branch_path.npz"
    initial = np.einsum("st,td->sd", matrix, problem.initial)
    np.savez_compressed(file, force_n=forces, rod_force_n=rods, support_load_n=support,
        support_quadrature_force_n=quadrature, displacement_m=initial,
        initialization_state_map=matrix, time_load_s=all_times, stage_labels=labels,
        q_dofs_microstrain=q, valid_direction_region=np.isfinite(q).all(axis=-1),
        **{"measured_"+key: value for key, value in measured_all.items()})
    network.extend_path(matrix)
    problem.initial = initial
    problem.boundary_path = file.relative_to(ROOT).as_posix()
    problem.boundary = dict(np.load(file))
    problem.force = p.tensor(forces)[None]
    problem.replay.set_boundary(rods, support)
    setup = copy.deepcopy(source["setup"])
    setup["boundary_arrays"] = problem.boundary_path
    setup.pop("regional_history_training", None)
    setup["representation"] = f"One shared material network predicts {len(forces)} chronological states with separate outputs at repeated forces. Physical history is recomputed over the complete sequence."
    setup["initialization"] = config["initialization"]
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    readout = FixedShuntReadout(uniform_parameters(), settings, p.tensor)
    query = setup["training_parameters"]+setup["query_parameters"]
    result = problem.evaluate(network, query, readout)
    with np.load(source_path.with_name(source_path.stem+"_query.npz")) as before:
        for key, value in result.items():
            rebuilt = value[:source_steps] if key == "force_n" else (value if key == "parameters" else value[:, :source_steps])
            np.testing.assert_allclose(rebuilt, before[key], rtol=1e-9, atol=1e-13)
    name = "loading_history_initial"
    torch.save(dict(network=network.state_dict()), OUTPUT/(name+".pt"))
    np.savez_compressed(OUTPUT/(name+"_query.npz"), **result)
    record = dict(name=name, setup=setup, segments=[], force_source=problem.boundary_path,
        initialization=dict(checkpoint=source_path.with_suffix(".pt").name,
                            state_map=problem.boundary_path, status="initialization_only"),
        interpretation="Prefix-preserving extrapolated initial guess. No high-load physical training, specimen fit, circuit reselection or branch prediction has been established.")
    (OUTPUT/(name+".json")).write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    polygon = json.loads((ROOT/"results/ideation/B2_local_boundary_correspondence.json").read_text())["moment_admissibility"]["trestle_polygon_m"]
    raw_rods = raw["rod_force_increment_kn"][:, mapping]+old["rod_force_n"][0]/1000
    reaction, margin = admissibility(raw["force_kn"], raw_rods, -p.gravity.numpy().sum()/1000,
                                    centres, eccentricity, np.asarray(polygon))
    target = time_s <= times[-1]
    bad = np.flatnonzero((reaction < 0) | (margin < 0))
    up = int(np.flatnonzero(labels == "rising_300kn")[0])
    down = len(labels)-1
    peak = int(np.flatnonzero(labels == "global_peak")[0])
    nominal = p.tensor([[1., -1.]])
    with torch.no_grad():
        state = network(nominal)
        full_force, _, _ = problem.equilibrium(state, nominal)
        intact_bulk_stress = stress_summary(p, state[0, peak].numpy())
        displacement_change = float((state[0, down]-state[0, up]).abs().max())
    support_resultant = quadrature.sum(axis=(1, 2))
    support_moment = np.einsum("sfq,fqc->sc", quadrature, points[..., :2])
    balance = support_resultant+forces-rods.sum(axis=1)+float(p.gravity.sum())
    moment_balance = support_moment+forces[:, None]*eccentricity+gravity_moment-rods@centres
    report = dict(configuration="inputs/loading_history_coverage.json", boundary_arrays=problem.boundary_path,
        network_record=f"results/ideation/stateful_pilot/fixed_reference_history/{name}.json",
        source_network=config["source_network"], source_prefix_steps=source_steps, physical_steps=len(forces),
        stage_labels=labels, time_load_s=all_times, force_kn=forces/1000,
        measured_local_mean_mm=measured_all["cone_relative_mean_increment_mm"],
        measured_machine_increment_mm=measured_all["jack_increment_mm"],
        optical_valid_direction_region=np.isfinite(q).all(axis=-1),
        control=config["control"], fixed_eccentricity_m=eccentricity,
        boundary_check=dict(native_mechanical_samples_to_second_300kn=int(target.sum()),
            support_force_range_kn=[reaction[target].min(), reaction[target].max()],
            inadmissible_samples_to_second_300kn=int(np.sum(target & ((reaction < 0) | (margin < 0)))),
            first_inadmissible_time_s=float(time_s[bad[0]]) if len(bad) else None,
            first_inadmissible_force_kn=float(raw["force_kn"][bad[0]]) if len(bad) else None,
            entire_record_minimum_conditional_support_kn=float(reaction.min()),
            maximum_selected_force_balance_n=float(abs(balance).max()),
            maximum_selected_moment_balance_n_m=float(abs(moment_balance).max())),
        same_force_branches=dict(step_indices=[up, down], force_kn=[forces[up]/1000, forces[down]/1000],
            measured_local_mean_mm=measured_all["cone_relative_mean_increment_mm"][[up, down]],
            measured_q_microstrain=q[[up, down]], initialized_max_displacement_difference_m=displacement_change,
            meaning="Distinct trainable output rows start from the same secant field. Their measured region/relative-motion difference is an unresolved target, not predicted by this initialization."),
        initialization_check=dict(material_queries=len(query), original_prefix_reconstructed=True,
            nominal_maximum_nodal_residual_by_step_n=full_force.abs().amax(dim=-1)[0].numpy(),
            nominal_peak_elastic_bulk_stress=intact_bulk_stress,
            maximum_damage_by_query=result["maximum_damage"][:, -1]),
        material_evidence=material_curve_evidence(),
        source_basis=["references/papers/jansson2024_dataset/full.md sections 4.3 and 4.5",
                      "https://pmc.ncbi.nlm.nih.gov/articles/PMC11460454/"],
        continuation=config["continuation"], elapsed_s=time.perf_counter()-started)
    output = ROOT/"results/ideation/B2_loading_history_coverage.json"
    output.write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), steps=len(forces), elapsed_s=report["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
