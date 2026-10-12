"""Trace a prescribed path in local relative displacement.

The actuator coordinate is solved together with the spatial field. A reversal
of that coordinate describes a branch fold, not the actual machine trajectory.
Only completed states enter the irreversible bulk and interface histories.
"""

import argparse
import json
import time

import numpy as np
import torch

from analyze_controlled_capacity import controlled_forces
from analyze_postpeak_branch import material_coordinates
from analyze_postpeak_deformation import responses
from controlled_branch_solver import CurrentControlledTangent, IndirectControlTangent
from controlled_fracture_history import ControlledFractureProblem, ROOT


class IndirectBalanceError(RuntimeError):
    """The trial field could not close its constituent/local constraints."""


@torch.no_grad()
def balance_indirect_terminal(problem, network, field, eta, machine, target):
    """Close constituent equilibrium and the local constraint simultaneously.

    The nineteenth correction is actuator displacement / sqrt(compliance).
    This scaling matches the existing energy-normalized rigid coordinates.
    The preceding fields and their damage are never re-solved.
    """
    balance = network.constituent_balance
    p = problem.physics
    local = p.relative_operator.mean(dim=0)
    local_modes = local@balance.modes
    root_compliance = np.sqrt(balance.compliance)
    offsets = balance.boundary_values(field[None])[0]
    jumps = offsets[:-1, :balance.sizes[0]].reshape(len(field)-1, -1, 3)
    compression = problem.base.materials(eta[None])[0, 1]
    prior = p.interface_path(jumps[None], compression)[2][0, -1]
    target0 = offsets[0, -1]+machine+balance.compliance*balance.reference_force
    local0 = local@(field[-1]-field[0])-target
    correction = field.new_zeros(19)

    def equations(value, tangent=False):
        result = balance.equations(value[:18], offsets[-1], prior, compression,
            target0+root_compliance*value[-1], tangent=tangent)
        residual = torch.cat([result[0], ((local0+local_modes@value[:18])/root_compliance)[None]])
        if not tangent:
            return residual
        matrix = field.new_zeros((19, 19))
        matrix[:18, :18] = result[2]
        matrix[:18, 18] = -balance.jack_modes/root_compliance
        matrix[18, :18] = local_modes/root_compliance
        return residual, matrix

    for _ in range(80):
        residual, matrix = equations(correction, tangent=True)
        if float(residual.abs().max()) < 1e-10:
            last = field[-1]+correction[:18]@balance.modes.T
            return torch.cat([field[:-1], last[None]]), float(machine+root_compliance*correction[-1])
        direction = torch.linalg.solve(matrix, -residual)
        value = float(residual@residual)
        scale = 1.
        for _ in range(50):
            trial = correction+scale*direction
            next_residual = equations(trial)
            if (float(next_residual@next_residual) <= (1-1e-4*scale)*value
                    or float(next_residual.abs().max()) < 1e-10):
                correction = trial
                break
            scale *= .5
        else:
            raise IndirectBalanceError("Indirect constituent equilibrium line search failed")
    raise IndirectBalanceError("Indirect constituent equilibrium did not converge")


def local_predictor(prefix, controls, local, target):
    coordinate = (prefix-prefix[:1])@local
    previous = float(coordinate[-1]-coordinate[-2])
    if previous == 0:
        raise ValueError("The local predictor requires distinct preceding local coordinates")
    ratio = (target-float(coordinate[-1]))/previous
    field = torch.cat([prefix, prefix[-1:]+ratio*(prefix[-1:]-prefix[-2:-1])])
    machine = controls[-1]+ratio*(controls[-1]-controls[-2])
    return field, float(machine)


def solve_local_stage(problem, network, baseline, eta, machine, target, *,
                      iterations, save_progress, previous=None, tolerance=.02,
                      reference_objective=None, linear_preconditioner="reference",
                      force_initial_update=False):
    p = problem.physics
    current = baseline.clone()
    local = p.relative_operator.mean(dim=0)
    compliance = network.compliance
    rows = [] if previous is None else list(previous["iterations"])

    @torch.no_grad()
    def evaluate(field, control):
        problem.machine[-1] = control
        force = controlled_forces(problem, field)
        residual, state = problem.equilibrium(field[None], force[None], eta[None])
        residual = residual[0, -1]
        whitened = problem.coordinates.whiten(residual[None, None]).ravel()
        return float(whitened@whitened), residual, force, state

    initial = evaluate(current, machine)[0]
    reference = initial if previous is None else previous["solver_reference_objective"]
    if reference_objective is not None:
        reference = max(reference, reference_objective)
        for row in rows:
            row["objective_fraction_of_reference"] = row["objective"]/reference

    def report(termination):
        return dict(target_local_increment_mm=target*1e3, machine_increment_mm=machine*1e3,
                    iterations=rows, termination=termination, solver_reference_objective=reference)

    if initial == 0 or (initial <= tolerance**2*reference
                       and not (force_initial_update and previous is None)):
        return report("residual_reduced"), current, machine

    for _ in range(iterations):
        started = time.perf_counter()
        value, residual, force, state = evaluate(current, machine)
        print(json.dumps(dict(event="assemble_indirect_tangent", iteration=len(rows)+1)), flush=True)
        tangent = CurrentControlledTangent(problem, current, eta)
        bordered = IndirectControlTangent(tangent, p.jack_operator.numpy(), local.numpy(), compliance)
        constraint = float(local@(current[-1]-current[0]))-target
        direction, linear = bordered.solve(np.r_[residual.numpy(), constraint/compliance], tolerance,
            preconditioner=linear_preconditioner)
        scale = 1.
        rejected = []
        for _ in range(30):
            trial = torch.cat([current[:-1], current[-1:]+scale*p.tensor(direction[:-1])[None]])
            trial_machine = machine+scale*direction[-1]
            try:
                trial, trial_machine = balance_indirect_terminal(problem, network, trial, eta, trial_machine, target)
            except torch.linalg.LinAlgError:
                rejected.append(dict(scale=scale, reason="singular_indirect_constituent_tangent"))
                scale *= .5
                continue
            except IndirectBalanceError as error:
                rejected.append(dict(scale=scale, reason=str(error)))
                scale *= .5
                continue
            next_value, _, next_force, next_state = evaluate(trial, trial_machine)
            if next_value <= value*(1-1e-4*scale):
                break
            scale *= .5
        else:
            problem.machine[-1] = machine
            return report("line_search_stalled"), current, machine
        np.testing.assert_array_equal(trial[:-1].numpy(), baseline[:-1].numpy())
        current, machine = trial, trial_machine
        row = dict(iteration=len(rows)+1, accepted_scale=scale, linear=linear,
            machine_increment_mm=machine*1e3, predicted_force_kn=float(next_force[-1]/1e3),
            local_mean_increment_mm=float(local@(current[-1]-current[0])*1e3),
            rod_total_kn=float(p.anchor_loads(current[-1]).sum()/1e3),
            objective=next_value, objective_fraction_of_reference=next_value/reference,
            seconds=time.perf_counter()-started, rejected_trials=rejected)
        rows.append(row)
        print(json.dumps(row), flush=True)
        save_progress(report("in_progress"), current, machine)
        if next_value <= tolerance**2*reference:
            return report("residual_reduced"), current, machine
    return report("iteration_limit"), current, machine


def reconstruct(problem, report, arrays, source, *, path_tangent=False):
    p = problem.physics
    field = p.tensor(arrays["displacement_m"])
    force = p.tensor(arrays["force_n"])
    eta = p.tensor(report["nominal_parameters"])
    count = len(source["displacement_m"])
    np.testing.assert_array_equal(field[:count].numpy(), source["displacement_m"])
    np.testing.assert_array_equal(arrays["machine_increment_m"][:count], source["machine_increment_m"][:count])
    np.testing.assert_allclose(force.numpy(), controlled_forces(problem, field).numpy(), rtol=1e-13, atol=1e-7)
    local = p.relative_operator.mean(dim=0)
    coordinates = ((field-field[:1])@local).numpy()
    for row in report["stages"]:
        np.testing.assert_allclose(coordinates[row["index"]]*1e3, row["target_local_increment_mm"], atol=1e-9)
    selected = sorted({count-1, count, int(np.argmax(force.numpy())),
                       int(np.argmax(arrays["machine_increment_m"])), len(field)-1})
    selected = [index for index in selected if index < len(field)]
    samples = []
    with torch.no_grad():
        _, state = problem.equilibrium(field[None], force[None], eta[None])
        for index in selected:
            response = responses(problem, field[None, :index+1], force[None, :index+1], eta[None])
            saved = next((row for row in report["stages"] if row["index"] == index), None)
            if saved is not None:
                for key, value in response.items():
                    np.testing.assert_allclose(value, saved["after"][key], rtol=1e-9, atol=1e-9)
            volume = p.tensor(problem.fracture.volumes)
            young = problem.base.materials(eta[None])[0, 0]
            knee = (problem.fracture.law.stress_knots_pa[1]/young
                    +problem.fracture.law.opening_knots_m[1]/problem.fracture.softening_length)
            area = (p.area[:, None]*p.surface_weights).flatten()
            samples.append(dict(index=index, machine_increment_mm=float(problem.machine[index]*1e3),
                **response,
                top_volume_past_first_knee_fraction=float(((state["bulk_history"][0, index] >= knee)*volume).sum()/volume.sum()),
                fully_debonded_interface_area_fraction=float(((state["interface_damage"][0, index] >= 1)*area).sum()/area.sum()),
                **material_coordinates(problem, field[:index+1], eta)))
    for key in ("bulk_history", "interface_history"):
        if not bool((state[key][:, 1:] >= state[key][:, :-1]).all()):
            raise AssertionError(f"The local continuation lost {key}")
    if path_tangent:
        slopes = []
        compliance = problem.config["apparatus"]["compliance_m_per_n"]
        for index in selected:
            tangent = CurrentControlledTangent(problem, field[:index+1], eta)
            bordered = IndirectControlTangent(tangent, p.jack_operator.numpy(), local.numpy(), compliance)
            rhs = np.zeros(p.ndof+1)
            rhs[-1] = -1/compliance
            direction, linear = bordered.solve(rhs)
            slopes.append(dict(index=index, machine_slope_mm_per_local_mm=float(direction[-1]),
                force_slope_kn_per_local_mm=float((direction[-1]-p.jack_operator.numpy()@direction[:-1])/compliance/1e6),
                local_slope=float(local.numpy()@direction[:-1]), linear=linear))
        report["local_path_tangents"] = slopes
    report["mechanism_samples"] = samples
    return dict(reconstructed_indices=selected, original_path_preserved=True,
                fixed_reference=True, exact_control=True, local_coordinate=True,
                causal_damage_history=True, constituent_force_balance=True, strip_endpoint_integral=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-targets-mm", type=float, nargs="+")
    parser.add_argument("--source", default="results/ideation/B2_controlled_capacity.json")
    parser.add_argument("--prefix-index", type=int,
        help="Last source state retained before starting a new local control history")
    parser.add_argument("--output", default="results/ideation/B2_controlled_turning.json")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-responses", action="store_true")
    parser.add_argument("--path-tangent", action="store_true")
    args = parser.parse_args()
    if args.path_tangent and not args.rebuild_responses:
        parser.error("Use --path-tangent with --rebuild-responses")
    if not args.local_targets_mm and not (args.resume or args.rebuild_responses):
        parser.error("Supply local displacement targets")
    torch.set_num_threads(2)
    started = time.perf_counter()
    output = ROOT/args.output
    source_path = ROOT/(json.loads(output.read_text())["source_path"]
        if args.resume or args.rebuild_responses else args.source)
    if source_path.resolve() == output.resolve():
        raise ValueError("The continued path must not overwrite its source")
    source_report = json.loads(source_path.read_text())
    source = dict(np.load(source_path.with_suffix(".npz")))
    prefix_index = (json.loads(output.read_text()).get("prefix_index")
        if args.resume or args.rebuild_responses else args.prefix_index)
    if prefix_index is not None:
        if not 0 < prefix_index < len(source["displacement_m"]):
            raise ValueError("Select a completed source prefix containing at least two states")
        source = {key: source[key][:prefix_index+1] for key in
                  ("displacement_m", "machine_increment_m", "force_n")}
    checkpoint = ROOT/source_report["checkpoint"]
    saved = json.loads(checkpoint.read_text())
    configuration = source_report.get("configuration", saved["configuration"])
    problem = ControlledFractureProblem(configuration)
    p = problem.physics
    network = problem.load_network(exact_control=True, fixed_reference=saved["fixed_reference"])
    network.load_state_dict(torch.load(checkpoint.with_suffix(".pt"), weights_only=True)["network"])
    eta = p.tensor(source_report["nominal_parameters"])
    prefix = p.tensor(source["displacement_m"])
    controls = source["machine_increment_m"][:len(prefix)].copy()
    candidate, previous = None, None
    report = dict(source_path=source_path.relative_to(ROOT).as_posix(), checkpoint=source_report["checkpoint"],
        nominal_parameters=eta.tolist(), stages=[], local_targets_mm=args.local_targets_mm,
        role="Indirect local-displacement continuation of the existing equilibrium branch; machine reachability is separate")
    if prefix_index is not None:
        report["prefix_index"] = prefix_index
    report["configuration"] = configuration
    if args.resume or args.rebuild_responses:
        report = json.loads(output.read_text())
        arrays = dict(np.load(output.with_suffix(".npz")))
        prefix, controls = p.tensor(arrays["displacement_m"]), arrays["machine_increment_m"]
        if "candidate_displacement_m" in arrays:
            candidate = p.tensor(arrays["candidate_displacement_m"])
            machine = float(arrays["candidate_machine_increment_m"])
            previous = report["current_result"]
        if args.local_targets_mm is not None:
            if candidate is not None:
                raise ValueError("New targets can only extend a completed local path")
            report["local_targets_mm"] += args.local_targets_mm
    problem.machine = p.tensor(controls)
    if args.rebuild_responses:
        report["checks"] = reconstruct(problem, report, arrays, source, path_tangent=args.path_tangent)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(report["checks"]), flush=True)
        return
    local = p.relative_operator.mean(dim=0)
    targets = np.asarray(report["local_targets_mm"])*1e-3
    origin = float(local@(p.tensor(source["displacement_m"][-1])-prefix[0]))
    if not np.isfinite(targets).all() or (np.diff(np.r_[origin, targets]) == 0).any():
        raise ValueError("Local targets must be finite with nonzero successive increments")
    prior_seconds = report.get("elapsed_seconds", 0.)
    for key in ("checks", "mechanism_samples", "local_path_tangents"):
        report.pop(key, None)

    def persist(result, field, control):
        report["current_result"] = result
        report["termination"] = result["termination"]
        report["elapsed_seconds"] = prior_seconds+time.perf_counter()-started
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        arrays = dict(displacement_m=prefix.numpy(), machine_increment_m=controls,
            force_n=controlled_forces(problem, prefix).numpy())
        if len(field) == len(prefix)+1:
            arrays.update(candidate_displacement_m=field.numpy(), candidate_machine_increment_m=np.asarray(control))
        np.savez_compressed(output.with_suffix(".npz"), **arrays)

    for target in targets[len(report["stages"]):]:
        print(json.dumps(dict(event="begin_local_stage", target_local_increment_mm=target*1e3)), flush=True)
        if candidate is None:
            candidate, machine = local_predictor(prefix, controls, local, target)
            candidate, machine = balance_indirect_terminal(problem, network, candidate, eta, machine, target)
        problem.machine = p.tensor(np.r_[controls, machine])
        result, candidate, machine = solve_local_stage(problem, network, candidate, eta, machine, target,
            iterations=args.iterations, save_progress=persist, previous=previous)
        with torch.no_grad():
            force = controlled_forces(problem, candidate)
            result["after"] = responses(problem, candidate[None], force[None], eta[None])
        persist(result, candidate, machine)
        if result["termination"] != "residual_reduced":
            return
        np.testing.assert_array_equal(candidate[:-1].numpy(), prefix.numpy())
        prefix = candidate
        controls = np.r_[controls, machine]
        report["stages"].append(dict(index=len(prefix)-1, **result))
        persist(dict(termination="local_targets_completed" if len(report["stages"]) == len(targets) else "in_progress"), prefix, machine)
        report.pop("current_result", None)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        candidate, previous = None, None
        print(json.dumps(dict(target_local_increment_mm=target*1e3, machine_increment_mm=machine*1e3,
            predicted_force_kn=float(force[-1]/1e3), termination=report["termination"])), flush=True)


if __name__ == "__main__":
    main()
