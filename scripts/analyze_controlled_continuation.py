"""Follow the recorded actuator sequence from the fixed neural reference.

Each acquisition carries the corrected preceding fields and their damage.
The saved shared increment is only a predictor; observed force, LVDT and fibre
strain are not field labels. Shared network weights are not changed.
"""

import argparse
import json
import time

import numpy as np
import torch

from analyze_postpeak_branch import solve_stage
from analyze_postpeak_deformation import responses
from controlled_fracture_history import ControlledFractureProblem, ROOT
from fixed_reference_history import OUTPUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-responses", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    checkpoint = OUTPUT/"controlled_fracture_flexure.json"
    saved = json.loads(checkpoint.read_text())
    problem = ControlledFractureProblem(saved["configuration"])
    p = problem.physics
    network = problem.load_network(exact_control=True, fixed_reference=saved["fixed_reference"])
    network.load_state_dict(torch.load(checkpoint.with_suffix(".pt"), weights_only=True)["network"])
    eta = p.tensor([1., -1.])
    with torch.no_grad():
        shared, _ = network(eta[None])
    shared = shared[0]
    labels = problem.boundary["stage_labels"].tolist()
    output = ROOT/"results/ideation/B2_controlled_continuation.json"
    if args.rebuild_responses:
        report = json.loads(output.read_text())
        arrays = dict(np.load(output.with_suffix(".npz")))
        field, force = p.tensor(arrays["displacement_m"]), p.tensor(arrays["force_n"])
        np.testing.assert_array_equal(field[0].numpy(), shared[0].numpy())
        np.testing.assert_array_equal(arrays["machine_increment_m"], problem.machine.numpy())
        selected = {"rising_300kn", "global_peak", "event_1_after", "event_2_after", "falling_300kn"}
        selected.add(report["stages"][-1]["stage"])
        checked = []
        with torch.no_grad():
            for row in report["stages"]:
                if row["stage"] not in selected:
                    continue
                count = row["index"]+1
                actual = responses(problem, field[None, :count], force[None, :count], eta[None])
                for key in actual:
                    np.testing.assert_allclose(actual[key], row["after"][key], rtol=1e-9, atol=1e-9)
                checked.append(row["stage"])
            _, state = problem.equilibrium(field[None], force[None], eta[None])
        for key in ("bulk_history", "interface_history"):
            if not bool((state[key][:, 1:] >= state[key][:, :-1]).all()):
                raise AssertionError(f"The sequential path lost {key}")
        for row in report["stages"]:
            row["reference_role"] = "Balanced acquisition predictor with the same corrected physical prefix"
            for iteration in row["iterations"]:
                if "objective_fraction_of_shared_reference" in iteration:
                    iteration["objective_fraction_of_reference"] = iteration.pop("objective_fraction_of_shared_reference")
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(dict(reconstructed_stages=checked, fixed_reference=True,
                              recorded_controls=True, causal_damage_history=True)), flush=True)
        return
    prefix = shared[:1].clone()
    candidate = None
    reference = prefix
    report = dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(), nominal_parameters=eta.tolist(),
        stages=[], termination="in_progress", current_stage=labels[1],
        role="Sequential physical correction in the shared PINN spatial field with causal history and recorded actuator increments; weights retained")
    if args.resume:
        report = json.loads(output.read_text())
        arrays = dict(np.load(output.with_suffix(".npz")))
        prefix = p.tensor(arrays["displacement_m"])
        np.testing.assert_array_equal(prefix[0].numpy(), shared[0].numpy())
        if len(prefix) == len(shared):
            print(json.dumps(dict(termination=report["termination"], already_complete=True)))
            return
        candidate = p.tensor(arrays["candidate_displacement_m"])
        reference = p.tensor(arrays["reference_displacement_m"])
    prior_seconds = report.get("elapsed_seconds", 0.)

    def persist(field):
        report["elapsed_seconds"] = prior_seconds+time.perf_counter()-started
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        force = torch.cat([prefix.new_tensor([problem.reference_force]),
            problem.reference_force+(problem.machine[1:len(prefix)]
                -(prefix[1:]-prefix[:1])@p.jack_operator)/network.compliance])
        arrays = dict(displacement_m=prefix.numpy(), force_n=force.numpy(),
                      stage_labels=np.asarray(labels), machine_increment_m=problem.machine.numpy())
        if len(prefix) < len(shared):
            arrays.update(candidate_displacement_m=field.numpy(), reference_displacement_m=reference.numpy())
        np.savez_compressed(output.with_suffix(".npz"), **arrays)

    for step in range(len(prefix), len(shared)):
        print(json.dumps(dict(stage=labels[step], index=step, event="begin_acquisition")), flush=True)
        previous_iterations = []
        previous_result = None
        if candidate is None or len(candidate) != step+1:
            candidate = torch.cat([prefix, prefix[-1:]+shared[step:step+1]-shared[step-1:step]])
            with torch.no_grad():
                candidate = network.balance(candidate[None], eta[None])[0]
            reference = candidate.clone()
        elif report.get("current_result"):
            previous_result = report["current_result"]
            previous_iterations = previous_result["iterations"]
        report.update(current_stage=labels[step], termination="in_progress")

        def save_stage(result, field, force):
            result = dict(result)
            result["iterations"] = previous_iterations+[
                dict(row, iteration=len(previous_iterations)+row["iteration"]) for row in result["iterations"]]
            if previous_result is not None:
                result["before"] = previous_result["before"]
                result["material_coordinates"] = dict(result["material_coordinates"],
                    before=previous_result["material_coordinates"]["before"])
            result["reference_role"] = "Balanced acquisition predictor with the same corrected physical prefix"
            for row in result["iterations"]:
                if "objective_fraction_of_shared_reference" in row:
                    row["objective_fraction_of_reference"] = row.pop("objective_fraction_of_shared_reference")
            report["current_result"] = result
            persist(field)

        result, candidate, _ = solve_stage(problem, network, candidate, eta,
            reference_field=reference, iterations=12, save_progress=save_stage)
        save_stage(result, candidate, None)
        if result["termination"] != "residual_reduced":
            report["termination"] = result["termination"]
            persist(candidate)
            print(json.dumps(dict(stage=labels[step], termination=report["termination"])), flush=True)
            return
        np.testing.assert_allclose(candidate[:-1].numpy(), prefix.numpy(), rtol=0, atol=2e-13)
        prefix = candidate
        report["stages"].append(dict(stage=labels[step], index=step, **report.pop("current_result")))
        report["current_stage"] = labels[step+1] if step+1 < len(shared) else None
        if len(prefix) == len(shared):
            report["termination"] = "recorded_path_completed"
        persist(candidate)
        print(json.dumps(dict(stage=labels[step], after=result["after"])), flush=True)
        candidate = None
    print(json.dumps(dict(saved=str(output), termination=report["termination"],
                          elapsed_seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
