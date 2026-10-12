"""Continue the corrected PINN spatial field beyond the recorded actuator path.

Only prescribed actuator motion is extended. The recorded prefix and all its
damage remain physical history; observation labels and shared weights are kept.
"""

import argparse
import json
import time

import numpy as np
import torch

from analyze_postpeak_branch import solve_stage
from analyze_postpeak_deformation import responses
from controlled_branch_solver import CurrentControlledTangent
from controlled_fracture_history import ControlledFractureProblem, ROOT


def loading_controls(recorded, end_m, step_m):
    """Append monotonically increasing controls without altering acquisitions."""
    recorded = np.asarray(recorded, dtype=float)
    if not np.isfinite([end_m, step_m]).all() or step_m <= 0 or end_m <= recorded[-1]:
        raise ValueError("The endpoint must exceed the recorded path and the step must be positive")
    later = np.arange(recorded[-1]+step_m, end_m, step_m)
    return np.r_[recorded, later, end_m]


def secant_predictor(prefix, controls):
    """Extrapolate the last completed physical increment to the next control."""
    index = len(prefix)
    previous = controls[index-1]-controls[index-2]
    if previous <= 0:
        raise ValueError("The capacity predictor requires a preceding positive control increment")
    ratio = (controls[index]-controls[index-1])/previous
    return torch.cat([prefix, prefix[-1:]+ratio*(prefix[-1:]-prefix[-2:-1])])


@torch.no_grad()
def balance_terminal(problem, network, field, eta):
    """Balance the new state using the fixed, complete physical prefix.

    Calling the full neural history layer would also resolve earlier states.
    Their accepted fields instead remain byte-identical during continuation.
    This primal correction uses the same constituent equations and material law.
    """
    balance = network.constituent_balance
    offsets = balance.boundary_values(field)
    jumps = offsets[:, :-1, :balance.sizes[0]].reshape(len(field), field.shape[1]-1, -1, 3)
    compression = problem.base.materials(eta)[:, 1]
    prior = problem.physics.interface_path(jumps, compression[:, None, None])[2][:, -1]
    target = offsets[:, 0, -1]+problem.machine[field.shape[1]-1]+balance.compliance*balance.reference_force
    correction, _, _ = balance.solve(offsets[:, -1], prior,
        compression, target)
    return torch.cat([field[:, :-1], field[:, -1:]+(correction@balance.modes.T)[:, None]], dim=1)


def controlled_forces(problem, field):
    later = problem.reference_force+(problem.machine[1:len(field)]
        -(field[1:]-field[:1])@problem.physics.jack_operator)/problem.config["apparatus"]["compliance_m_per_n"]
    return torch.cat([field.new_tensor([problem.reference_force]), later])


def reconstruct(problem, report, arrays, source, *, path_tangent=False):
    """Rebuild the turning point and endpoint, and verify causal path storage."""
    p = problem.physics
    field, force = p.tensor(arrays["displacement_m"]), p.tensor(arrays["force_n"])
    count = len(source["displacement_m"])
    np.testing.assert_array_equal(field[:count].numpy(), source["displacement_m"])
    np.testing.assert_array_equal(arrays["machine_increment_m"][:count], source["machine_increment_m"])
    np.testing.assert_allclose(force.numpy(), controlled_forces(problem, field).numpy(), rtol=1e-13, atol=1e-7)
    eta = p.tensor(report["nominal_parameters"])[None]
    selected = {int(np.argmax(force.numpy())), len(field)-1, count}
    checked = []
    trial_checks = []
    with torch.no_grad():
        for row in report["stages"]:
            if row["index"] not in selected:
                continue
            end = row["index"]+1
            actual = responses(problem, field[None, :end], force[None, :end], eta)
            for key, value in actual.items():
                np.testing.assert_allclose(value, row["after"][key], rtol=1e-9, atol=1e-9)
            checked.append(row["stage"])
        _, state = problem.equilibrium(field[None], force[None], eta)
        if "candidate_displacement_m" in arrays:
            candidate = p.tensor(arrays["candidate_displacement_m"])
            np.testing.assert_array_equal(candidate[:-1].numpy(), field.numpy())
            actual = responses(problem, candidate[None], controlled_forces(problem, candidate)[None], eta)
            for key, value in actual.items():
                np.testing.assert_allclose(value, report["current_result"]["after"][key], rtol=1e-9, atol=1e-9)
            trial_checks.append(report["current_stage"])
        for row in report.get("refinements", []):
            index = row["prefix_length"]
            candidate = torch.cat([field[:index], p.tensor(arrays[row["fields"]+"_candidate_m"])[None]])
            original = problem.machine[index].clone()
            problem.machine[index] = row["machine_increment_mm"]*1e-3
            actual = responses(problem, candidate[None], controlled_forces(problem, candidate)[None], eta)
            problem.machine[index] = original
            for key, value in actual.items():
                np.testing.assert_allclose(value, row["result"]["after"][key], rtol=1e-9, atol=1e-9)
            trial_checks.append(row["fields"])
    for key in ("bulk_history", "interface_history"):
        if not bool((state[key][:, 1:] >= state[key][:, :-1]).all()):
            raise AssertionError(f"The continued path lost {key}")
    # The extent of released transfer distinguishes a growing damaged region
    # from a single quadrature point reaching a constitutive knee.
    volume = p.tensor(problem.fracture.volumes)
    area = (p.area[:, None]*p.surface_weights).flatten()
    young = problem.base.materials(eta)[0, 0]
    knee = problem.fracture.law.stress_knots_pa[1]/young+problem.fracture.law.opening_knots_m[1]/problem.fracture.softening_length
    mechanism_rows = []
    for index in sorted(selected | {count-1}):
        if index >= len(field):
            continue
        mechanism_rows.append(dict(index=index, machine_increment_mm=float(problem.machine[index]*1e3),
            force_kn=float(force[index]/1e3),
            maximum_equivalent_opening_mm=float(state["bulk_equivalent_opening_m"][0, index].max()*1e3),
            top_volume_past_first_knee_fraction=float(((state["bulk_history"][0, index] >= knee)*volume).sum()/volume.sum()),
            fully_debonded_interface_area_fraction=float(((state["interface_damage"][0, index] >= 1)*area).sum()/area.sum())))
    report["mechanism_samples"] = mechanism_rows
    if path_tangent:
        compliance = problem.config["apparatus"]["compliance_m_per_n"]
        tangent = CurrentControlledTangent(problem, field, eta[0])
        direction, linear = tangent.solve(-p.jack_operator.numpy()/compliance)
        report["terminal_control_tangent"] = dict(index=len(field)-1,
            machine_increment_mm=float(problem.machine[len(field)-1]*1e3),
            force_slope_kn_per_mm=float((1-p.jack_operator.numpy()@direction)/compliance/1e6),
            local_increment_slope_mm_per_mm=float((p.relative_operator.numpy()@direction).mean()),
            linear=linear,
            role="Differential response of the last accepted equilibrium at fixed prior history; not a stability test")
    return dict(reconstructed_stages=checked, reconstructed_unfinished_trials=trial_checks,
                recorded_fields_preserved=True,
                fixed_reference=True, exact_control=True, causal_damage_history=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-mm", type=float)
    parser.add_argument("--step-mm", type=float, default=.25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--subdivide-current", action="store_true")
    parser.add_argument("--rebuild-responses", action="store_true")
    parser.add_argument("--path-tangent", action="store_true")
    args = parser.parse_args()
    if args.end_mm is None and not (args.resume or args.rebuild_responses):
        parser.error("Supply the extended actuator endpoint with --end-mm")
    if args.subdivide_current and (not args.resume or args.rebuild_responses):
        parser.error("Use --subdivide-current with --resume for an unfinished control step")
    if args.path_tangent and not args.rebuild_responses:
        parser.error("Use --path-tangent with --rebuild-responses")
    torch.set_num_threads(2)
    started = time.perf_counter()
    source_path = ROOT/"results/ideation/B2_controlled_continuation.json"
    source_report = json.loads(source_path.read_text())
    if source_report["termination"] != "recorded_path_completed":
        raise ValueError("Complete the recorded path before extending its loading")
    source = dict(np.load(source_path.with_suffix(".npz")))
    checkpoint = ROOT/source_report["checkpoint"]
    saved = json.loads(checkpoint.read_text())
    problem = ControlledFractureProblem(saved["configuration"])
    p = problem.physics
    network = problem.load_network(exact_control=True, fixed_reference=saved["fixed_reference"])
    network.load_state_dict(torch.load(checkpoint.with_suffix(".pt"), weights_only=True)["network"])
    eta = p.tensor(source_report["nominal_parameters"])
    output = ROOT/"results/ideation/B2_controlled_capacity.json"
    prefix = p.tensor(source["displacement_m"])
    candidate = None
    refinement_fields = {}
    report = dict(source_path=source_path.relative_to(ROOT).as_posix(),
        checkpoint=source_report["checkpoint"], nominal_parameters=eta.tolist(), stages=[],
        endpoint_mm=args.end_mm, step_mm=args.step_mm, termination="in_progress",
        role="Physical loading continuation in the shared PINN spatial field; recorded prefix and network weights retained")
    if args.resume or args.rebuild_responses:
        report = json.loads(output.read_text())
        arrays = dict(np.load(output.with_suffix(".npz")))
        refinement_fields = {key: value for key, value in arrays.items() if key.startswith("refinement_")}
        prefix = p.tensor(arrays["displacement_m"])
        np.testing.assert_array_equal(prefix[:len(source["displacement_m"])], source["displacement_m"])
        controls = arrays["machine_increment_m"]
        if args.end_mm is not None and args.end_mm != report["endpoint_mm"]:
            if len(prefix) != len(controls) or args.end_mm <= controls[-1]*1e3:
                raise ValueError("Only a completed path can be extended to a later endpoint")
            controls = loading_controls(controls, args.end_mm*1e-3, args.step_mm*1e-3)
            report.update(endpoint_mm=args.end_mm, step_mm=args.step_mm)
        if "candidate_displacement_m" in arrays:
            candidate = p.tensor(arrays["candidate_displacement_m"])
            reference = p.tensor(arrays["reference_displacement_m"])
            if len(candidate) != len(prefix)+1:
                candidate = None
        if args.subdivide_current:
            if candidate is None or "current_result" not in report:
                raise ValueError("There is no saved current trial to subdivide")
            index = len(prefix)
            midpoint = (controls[index-1]+controls[index])/2
            if not controls[index-1] < midpoint < controls[index]:
                raise ValueError("The control interval has no representable midpoint")
            refinements = report.setdefault("refinements", [])
            name = f"refinement_{len(refinements)+1:03d}"
            refinement_fields[name+"_candidate_m"] = candidate[-1].numpy()
            refinement_fields[name+"_reference_m"] = reference[-1].numpy()
            result = report.pop("current_result")
            result["termination"] = "subdivided_after_shortened_steps"
            refinements.append(dict(prefix_length=index, machine_increment_mm=controls[index]*1e3,
                inserted_increment_mm=midpoint*1e3, fields=name,
                result=result,
                reason="Refine the control increment after shortened Newton steps alternate local displacement"))
            controls = np.insert(controls, index, midpoint)
            candidate = None
    else:
        controls = loading_controls(source["machine_increment_m"], args.end_mm*1e-3, args.step_mm*1e-3)
    # The network is used only for its unchanged constitutive balance. The
    # physical problem owns new controls; its trained time output is not resized.
    problem.machine = p.tensor(controls)
    if args.rebuild_responses:
        checks = reconstruct(problem, report, arrays, source, path_tangent=args.path_tangent)
        report["checks"] = checks
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(checks), flush=True)
        if args.path_tangent:
            print(json.dumps(report["terminal_control_tangent"]), flush=True)
        return
    prior_seconds = report.get("elapsed_seconds", 0.)
    if len(prefix) < len(controls):
        report.pop("checks", None)
        report.pop("mechanism_samples", None)
        report.pop("terminal_control_tangent", None)
    labels = list(source["stage_labels"])+[f"extension_{i:03d}" for i in range(1, len(controls)-len(source["stage_labels"])+1)]
    rebalance = lambda field, eta: balance_terminal(problem, network, field, eta)

    def persist(field):
        report["elapsed_seconds"] = prior_seconds+time.perf_counter()-started
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        arrays = dict(displacement_m=prefix.numpy(), force_n=controlled_forces(problem, prefix).numpy(),
            stage_labels=np.asarray(labels), machine_increment_m=controls)
        arrays.update(refinement_fields)
        if len(prefix) < len(controls) and len(field) == len(prefix)+1:
            arrays.update(candidate_displacement_m=field.numpy(), reference_displacement_m=reference.numpy())
        np.savez_compressed(output.with_suffix(".npz"), **arrays)

    for step in range(len(prefix), len(controls)):
        print(json.dumps(dict(event="begin_extension", stage=labels[step], machine_increment_mm=controls[step]*1e3)), flush=True)
        previous = report.get("current_result") if candidate is not None else None
        if candidate is None:
            candidate = rebalance(secant_predictor(prefix, controls)[None], eta[None])[0]
            reference = candidate.clone()
        report.update(current_stage=labels[step], termination="in_progress")

        def save_stage(result, field, force):
            result = dict(result)
            if previous is not None:
                result["iterations"] = previous["iterations"]+[
                    dict(row, iteration=len(previous["iterations"])+row["iteration"]) for row in result["iterations"]]
                result["before"] = previous["before"]
                result["material_coordinates"] = dict(result["material_coordinates"], before=previous["material_coordinates"]["before"])
            result["reference_role"] = "Balanced secant predictor with the identical completed physical prefix"
            for row in result["iterations"]:
                if "objective_fraction_of_shared_reference" in row:
                    row["objective_fraction_of_reference"] = row.pop("objective_fraction_of_shared_reference")
            report["current_result"] = result
            persist(field)

        result, candidate, _ = solve_stage(problem, network, candidate, eta, reference_field=reference,
            iterations=12, save_progress=save_stage, rebalance=rebalance)
        save_stage(result, candidate, None)
        if result["termination"] != "residual_reduced":
            report["termination"] = result["termination"]
            persist(candidate)
            print(json.dumps(dict(stage=labels[step], termination=report["termination"])), flush=True)
            return
        np.testing.assert_array_equal(candidate[:-1].numpy(), prefix.numpy())
        prefix = candidate
        report["stages"].append(dict(stage=labels[step], index=step,
            machine_increment_mm=controls[step]*1e3, **report.pop("current_result")))
        report["current_stage"] = labels[step+1] if step+1 < len(controls) else None
        if len(prefix) == len(controls):
            report["termination"] = "extended_path_completed"
        persist(candidate)
        print(json.dumps(dict(stage=labels[step], after=result["after"])), flush=True)
        candidate = None
    print(json.dumps(dict(saved=str(output), termination=report["termination"], elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
