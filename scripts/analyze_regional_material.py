"""Separate material mechanisms under the saved complete force history."""

import argparse
import copy
import json
import time

import numpy as np
import torch

from analyze_asymmetric_boundary import FullEquilibriumSolver, load_problem
from boundary_history import ForceHistoryProblem, finite_arm_probes
from calibrate_support_response import json_values, local_invariants
from force_history_solver import solve_history_step
from initial_state_model import InitialStatePhysics, material_densities
from stateful_physics import ROOT


def case_problem(base, source, parameters, reference, gauge_stiffness):
    settings = copy.deepcopy(base.settings)
    settings["bulk"]["top_young_pa"] = parameters["top_young_pa"]
    settings["interface"]["compression_stiffness_pa_per_m"] = parameters["compression_stiffness_pa_per_m"]
    p = InitialStatePhysics(base.mesh, settings, base.config,
                            material_densities(base.config["weight"])[0])
    p.jack_operator = p.tensor(source["jack_operator"])
    replay = ForceHistoryProblem(p, reference, gauge_stiffness)
    return p, replay


def cone_interface_mean(p):
    """Mean vertical top-minus-cone jump over their bonded footprint."""
    cone_nodes = np.unique(p.mesh["tetra"][p.mesh["body"] == 2])
    selected = np.all(np.isin(p.mesh["interface_lower"], cone_nodes), axis=1)
    area = p.mesh["interface_area"][selected]
    mean = area[:, None]*np.einsum("q,qa->a", p.surface_weights.numpy(), p.shape.numpy())[None]/area.sum()
    operator = np.zeros(p.ndof)
    np.add.at(operator, 3*p.mesh["interface_upper"][selected].ravel()+2, mean.ravel())
    np.add.at(operator, 3*p.mesh["interface_lower"][selected].ravel()+2, -mean.ravel())
    return operator


def stress_summary(p, state):
    """Principal tension at physical integration points, weighted by volume."""
    values = state.reshape(-1, 3)[p.mesh["tetra"]].reshape(len(p.mesh["tetra"]), -1)
    strain = np.einsum("eqij,ej->eqi", p.b, values)
    young = np.where(p.mesh["body"] == 0, p.settings["bulk"]["top_young_pa"],
                     p.settings["bulk"]["substrate_young_pa"])
    nu = p.settings["bulk"]["poisson"]
    lame, mu = young*nu/((1+nu)*(1-2*nu)), young/(2*(1+nu))
    normal = 2*mu[:, None, None]*strain[..., :3]+lame[:, None, None]*strain[..., :3].sum(axis=-1, keepdims=True)
    stress = np.zeros((*strain.shape[:2], 3, 3))
    for i in range(3):
        stress[..., i, i] = normal[..., i]
    for component, (i, j) in enumerate([(0, 1), (1, 2), (0, 2)], start=3):
        stress[..., i, j] = stress[..., j, i] = mu[:, None]*strain[..., component]
    principal = np.linalg.eigvalsh(stress)[..., -1]/1e6
    rows = []
    for body, label in enumerate(["top", "substrate", "cone"]):
        selected = p.mesh["body"] == body
        value = principal[selected].ravel()
        weight = np.broadcast_to(p.volume[selected, None]/principal.shape[1], principal[selected].shape).ravel()
        order = np.argsort(value)
        cumulative = np.cumsum(weight[order])/weight.sum()
        rows.append(dict(body=label, principal_tension_volume_quantiles_mpa=
                         np.interp([.5, .95, .99], cumulative, value[order]),
                         maximum_principal_tension_mpa=value.max()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", help="Run named cases and retain the other completed cases")
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    config_path = ROOT/"inputs/regional_material_response.json"
    config = json.loads(config_path.read_text())
    boundary_config = json.loads((ROOT/config["boundary_configuration"]).read_text())
    asymmetric_config = json.loads((ROOT/boundary_config["asymmetric_configuration"]).read_text())
    asymmetric_report = json.loads((ROOT/boundary_config["asymmetric_report"]).read_text())
    selected = next(row for row in asymmetric_report["cases"] if row["name"] == boundary_config["reference_case"])
    source = dict(np.load(ROOT/selected["arrays"]))
    saved = dict(np.load(ROOT/config["boundary_arrays"]))
    base, _, _ = load_problem(asymmetric_config)
    base.jack_operator = base.tensor(source["jack_operator"])
    solver = FullEquilibriumSolver(base, source["displacement_m"][4])
    angles = np.concatenate([saved["azimuth_deg"], saved["azimuth_deg"]+120, saved["azimuth_deg"]+240])
    arm, _, _ = finite_arm_probes(base, angles, boundary_config["lvdt_arm_length_m"])
    cone_jump = cone_interface_mean(base)
    destination = ROOT/"data/processed/regional_material_response"
    destination.mkdir(parents=True, exist_ok=True)
    output = ROOT/"results/ideation/B2_regional_material_response.json"
    report = dict(configuration=config_path.relative_to(ROOT).as_posix(),
                  source_boundary_arrays=config["boundary_arrays"], cases=[],
                  parameter_basis=config["parameter_basis"], continuation=config["continuation"])
    if args.case:
        names = {row["name"] for row in config["cases"]}
        if not set(args.case) <= names:
            raise ValueError(f"Unknown case: {set(args.case)-names}")
        if output.exists():
            report["cases"] = [row for row in json.loads(output.read_text())["cases"] if row["name"] not in args.case]
    baseline_closure = -(saved["displacement_m"]-saved["displacement_m"][0])@cone_jump*1000
    report["baseline"] = dict(central_host_q_microstrain=saved["host_increment_microstrain"][-1, :, 1],
                               mean_local_motion_mm=saved["finite_arm_increment_mm"][-1].mean(),
                               mean_cone_interface_closure_mm=baseline_closure[-1],
                               stress_at_100kn=stress_summary(base, saved["displacement_m"][-1]))
    for parameters in config["cases"]:
        if args.case and parameters["name"] not in args.case:
            continue
        p, replay = case_problem(base, source, parameters, saved["displacement_m"][0],
                                 asymmetric_config["conditional_rod_stiffness_n_per_m"])
        states, updates = [], []
        for i, force in enumerate(saved["force_n"]):
            replay.set_boundary(saved["rod_force_n"][i], saved["support_load_n"][i])
            initial = saved["displacement_m"][i] if not states else states[-1]
            state, count, current_damage = solve_history_step(replay, force, states, initial, solver.preconditioner)
            states.append(state)
            updates.append(count)
            print(json.dumps(dict(case=parameters["name"], force_kn=force/1000,
                                  newton_updates=count, maximum_damage=current_damage,
                                  elapsed_s=time.perf_counter()-start)), flush=True)
        states = np.asarray(states)
        replay.set_boundary(saved["rod_force_n"], saved["support_load_n"])
        residual, damage, history = replay.equilibrium(p.tensor(states)[None], p.tensor(saved["force_n"])[None])
        delta = states-states[0]
        host = p.observations(p.tensor(delta))[0].numpy()*1e6
        local = (delta@arm.T).reshape(len(states), 3, -1).transpose(0, 2, 1)*1000
        closure = -delta@cone_jump*1000
        invariants = local_invariants(local)
        file = destination/(parameters["name"]+".npz")
        np.savez_compressed(file, force_n=saved["force_n"], displacement_m=states,
                            host_increment_microstrain=host, finite_arm_increment_mm=local,
                            mean_cone_interface_closure_mm=closure,
                            gauge_reference=replay.gauge_reference.numpy())
        report["cases"].append(dict(name=parameters["name"], parameters=parameters,
            arrays=file.relative_to(ROOT).as_posix(), force_kn=saved["force_n"]/1000,
            central_host_q_microstrain=host[:, :, 1],
            host_change_from_baseline_microstrain=host-saved["host_increment_microstrain"],
            mean_local_motion_mm=local.mean(axis=(1, 2)),
            local_mean_range_mm=np.stack([invariants[..., 0].min(axis=1), invariants[..., 0].max(axis=1)], axis=-1),
            local_amplitude_range_mm=np.stack([invariants[..., 1].min(axis=1), invariants[..., 1].max(axis=1)], axis=-1),
            mean_cone_interface_closure_mm=closure,
            local_minus_mean_interface_closure_mm=local.mean(axis=(1, 2))-closure,
            maximum_nodal_residual_n=float(residual.abs().max()), maximum_damage=float(damage.max()),
            maximum_history_m=float(history.max()), newton_updates=updates,
            maximum_gauge_reaction_n=float(replay.gauge_reactions(p.tensor(states)).abs().max()),
            stress_at_100kn=stress_summary(p, states[-1])))
        report["elapsed_s"] = time.perf_counter()-start
        output.write_text(json.dumps(json_values(report), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), elapsed_s=report["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
