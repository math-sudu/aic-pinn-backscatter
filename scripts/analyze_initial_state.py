"""Build and retain the initial equilibrium used by experimental calibration."""

import json
import time

import numpy as np
import torch

from initial_state_model import build_initial_problem, solve_intact
from spatial_resolution import symmetric_projection
from stateful_physics import ROOT


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    p, records, groups = build_initial_problem()
    # This diagnostic has symmetric density, supports and equal rod inputs.
    # Every convergence check still evaluates the full three-dimensional force.
    if np.ptp(p.config["anchors"]["stiffness_n_per_m"]) or np.ptp(p.config["anchors"]["assembly_tension_n"]):
        raise ValueError("Use full coordinates for unequal anchors")
    projection = symmetric_projection(p)
    destination = ROOT / "data/processed/initial_state_model"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "mesh.npz", **p.mesh)
    alignment = json.loads((ROOT / "results/ideation/B2_engineering_alignment.json").read_text(encoding="utf-8"))["cl100h"]
    reference_channels = dict(zip(alignment["mechanical_reference_channels"], alignment["mechanical_reference_values"]))
    reference_force = -reference_channels["Ch 1 Force"] * 1000
    forces = [0.0, reference_force, *p.config["diagnostic_force_n"]]
    states, iterations = [], []
    print(json.dumps(dict(ndof=p.ndof, concrete_weight_n=-float(p.gravity.sum()))), flush=True)
    for index, force in enumerate(forces):
        state, count = solve_intact(p, force, None if index == 0 else states[-1], projection)
        if index == 0:
            p.seat_anchors(p.tensor(state))
        states.append(state)
        iterations.append(count)
        print(json.dumps(dict(force_n=force, iterations=count, elapsed_s=time.perf_counter()-started)), flush=True)
    displacement = p.tensor(np.stack(states))[None]
    residual, damage, history = p.equilibrium(displacement, p.tensor([forces]))
    outputs = {name: value[0].numpy() for name, value in p.mechanical_outputs(displacement).items()}
    host, _ = p.observations(displacement)
    arrays = dict(displacement_m=displacement[0].numpy(), jack_force_n=np.asarray(forces),
                  anchor_reference_m=p.anchor_reference.numpy(), gravity_force_n=p.gravity.numpy(),
                  local_probe_points_m=p.probe_points, free_residual_n=residual[0].numpy(),
                  damage=damage[0].numpy(), history_m=history[0].numpy(),
                  host_increment_from_known_load_microstrain=(host[0]-host[0, 1]).numpy()*1e6,
                  **outputs)
    np.savez_compressed(destination / "equilibrium.npz", **arrays)
    weight = -float(p.gravity.sum())
    balance = np.asarray(forces) + outputs["trestle_force_n"] - outputs["rod_force_n"].sum(axis=1) - weight
    measured = np.load(ROOT / alignment["history_file"])
    peak = int(measured["force_kn"].argmax())
    release = -measured["inferred_trestle_force_change_kn"]
    index = int(release.argmax())
    # Same geometric volume, changing only the unresolved substrate cast group.
    body_volume = np.array([p.volume[p.mesh["body"] == body].sum() for body in (0, 1, 2)])
    weight_by_cast = {prefix: float((body_volume[0]*groups[p.config["weight"]["top_cube_prefix"]]
                                    + body_volume[1:].sum()*groups[prefix]) * p.config["weight"]["gravity_m_per_s2"])
                      for prefix in p.config["weight"]["substrate_cube_prefixes"]}
    stage_results = []
    for i, force in enumerate(forces):
        stage_results.append(dict(jack_force_n=force, trestle_force_n=float(outputs["trestle_force_n"][i]),
            rod_force_n=outputs["rod_force_n"][i].tolist(),
            local_relative_increment_from_known_load_mm=((outputs["local_relative_m"][i]-outputs["local_relative_m"][1])*1000).tolist(),
            jack_face_increment_from_known_load_mm=float((outputs["jack_face_mean_m"][i]-outputs["jack_face_mean_m"][1])*1000),
            vertical_balance_n=float(balance[i]), contact_newton_updates=iterations[i]))
    report = dict(
        purpose=p.config["purpose"], configuration="inputs/initial_state_model.json",
        mesh_file="data/processed/initial_state_model/mesh.npz",
        equilibrium_file="data/processed/initial_state_model/equilibrium.npz",
        concrete_weight_n=weight, concrete_weight_by_substrate_cast_n=weight_by_cast,
        body_volume_m3=body_volume.tolist(), cube_density_groups_kg_per_m3=groups, cube_records=records,
        resolved_settings=p.settings,
        assembly_definition="Equilibrate gravity and specified assembly rod tensions at zero jack load; seat spring reference lengths at that state. The following known-load reference is solved with those lengths unchanged.",
        known_load_reference_force_n=reference_force,
        load_control="Only the measured jack resultant is prescribed. The computed patch displacement excludes actuator/frame compliance. Rod forces, local displacements and optical outputs remain responses to fit or check.",
        observation_definition=p.config["local_observations"], probe_points_m=p.probe_points.tolist(),
        source_conflict=p.config["anchors"]["source_conflict"],
        stages=stage_results, maximum_nodal_equilibrium_residual_n=float(residual.abs().max()),
        maximum_initialization_damage=float(damage.max()), elapsed_s=time.perf_counter()-started,
        measured_support_constraints=dict(
            maximum_prepeak_release_kn=float(release[:peak].max()),
            maximum_recorded_release_kn=float(release[index]), time_of_maximum_release_s=float(measured["time_load_s"][index]),
            minimum_extra_initial_downward_load_if_entire_record_quasistatic_kn=float(release[index]-(weight-reference_force)/1000),
            meaning="Concrete self-weight can supply the observed peak-rising support-release scale. Across the entire record, the maximum inferred release exceeds this weight. Absolute rod/jack tare, fixture weight, pretension and dynamic transients remain unresolved; the excess is a conditional balance bound, not identified pretension."),
        continuation="B2/engineering_alignment/support_calibration: use this initialized residual and the saved mechanical/optical history to calibrate support/load-transfer correspondence before PINN method development; register real LVDT positions and separate jack compliance from local deformation."
    )
    output = ROOT / "results/ideation/B2_initial_state_model.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(result=output.relative_to(ROOT).as_posix(), weight_n=weight,
                          final_support_n=outputs["trestle_force_n"][-1], balance_n=balance.tolist())), flush=True)


if __name__ == "__main__":
    main()
