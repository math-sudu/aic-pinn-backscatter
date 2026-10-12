"""Assess a joint control-history checkpoint and an untrained reversal.

Saved physical fields enter this response assessment only. Training constructs
its targets from its own causal physical residual and current neural fields.
"""

import argparse
import json

import numpy as np
import torch

from analyze_control_history import compare_history
from controlled_fracture_history import ROOT
from train_control_history import SOURCE, load_pilot, responses, summary


def evaluate_reference(problem, network, readout, controls, values, reference):
    physical = np.load(reference.with_suffix(".npz"))
    fields = problem.physics.tensor(physical["displacement_m"])
    count = len(controls)
    np.testing.assert_array_equal(values["displacement_m"][:30], fields[:30])
    np.testing.assert_allclose(((fields[30:30+count]-fields[:1])@network.balance.local)*1e3,
                              controls, rtol=0, atol=1e-8)
    host = ((fields-fields[:1])@problem.base.functionals.q_matrix.T).numpy()*1e6
    raw = fields@problem.base.raw_matrix.T
    circuit = readout(raw).flatten(1)
    circuit = (circuit-circuit[:1]).numpy()*1e6
    actual, expected, error = compare_history(values, host, circuit)
    return dict(controls_mm=controls, summary=summary(values),
        physical_evaluation=reference.relative_to(ROOT).as_posix(),
        physical_gauge_abs_max_microstrain=float(raw.abs().max()*1e6),
        physical_tail_force_kn=(physical["force_n"][30:30+count]/1e3).tolist(),
        predicted_history=actual, physical_history=expected,
        history_contrast_error_rms_microstrain=error)


def between_protocols(left, right):
    result = {}
    for source in ("predicted_history", "physical_history"):
        parts = {name: np.asarray(right[source][name+"_microstrain"])
                       -np.asarray(left[source][name+"_microstrain"])
                 for name in ("host", "circuit")}
        parts["discrepancy"] = parts["circuit"]-parts["host"]
        result[source] = {name+"_microstrain": value.tolist() for name, value in parts.items()}
        result[source].update({name+"_rms_microstrain": float(np.sqrt(np.mean(value**2)))
                               for name, value in parts.items()})
    result["error_rms_microstrain"] = {name: float(np.sqrt(np.mean((
        np.asarray(result["predicted_history"][name+"_microstrain"])
        -result["physical_history"][name+"_microstrain"])**2)))
        for name in ("host", "circuit", "discrepancy")}
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-reference", required=True,
                        help="Completed local field protocol at the held 0.75-mm reversal")
    args = parser.parse_args()
    checkpoint = ROOT/args.checkpoint
    record = json.loads(checkpoint.read_text())
    if record["status"] != "training_segment_completed":
        raise ValueError("Protocol assessment requires a completed neural update")
    controls_mm = record["protocol_spec"]["controls_mm"]
    problem, network, eta, prefix, prefix_forces, _, _, _, readout, _ = load_pilot()
    network.load_state_dict(torch.load(checkpoint.with_suffix(".pt"), weights_only=True)["network"])
    saved = dict(np.load(checkpoint.with_suffix(".npz")))
    controls = problem.physics.tensor(controls_mm)*1e-3
    rebuilt = responses(problem, network, eta.expand(len(controls), -1),
        prefix.expand(len(controls), -1, -1), prefix_forces.expand(len(controls), -1),
        controls, readout, batched=True)
    for name, value in rebuilt.items():
        np.testing.assert_allclose(value, saved[name], rtol=1e-8, atol=1e-9)
    references = {0.8: SOURCE, 0.7: ROOT/"results/ideation/B2_control_history_lower_peak.json"}
    training = [evaluate_reference(problem, network, readout, protocol,
                    {name: value[index] for name, value in saved.items()}, references[protocol[1]])
                for index, protocol in enumerate(controls_mm)]
    query_controls = controls.new_tensor([[.6, .75, .6]])*1e-3
    query = responses(problem, network, eta, prefix, prefix_forces, query_controls, readout)
    query_reference = ROOT/args.query_reference
    if json.loads(query_reference.read_text())["termination"] != "local_targets_completed":
        raise ValueError("The held protocol evaluation is not complete")
    query_row = evaluate_reference(problem, network, readout, [.6, .75, .6], query, query_reference)
    query_row["role"] = "Untrained 0.75-mm reversal; same weights, physical fields used only for evaluation"
    report = dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        initialization_checkpoint=record["initialization_checkpoint"],
        role="Joint 0.7/0.8-mm protocol training with an untrained intermediate reversal",
        saved_fields_reconstructed=True, fixed_prefix_preserved=True,
        training_protocols=training, held_protocol=query_row,
        protocol_change=between_protocols(training[1], training[0]))
    output = checkpoint.with_name(checkpoint.stem+"_response.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(training=[row["history_contrast_error_rms_microstrain"] for row in training],
        held=query_row["history_contrast_error_rms_microstrain"],
        protocol_change=report["protocol_change"]["error_rms_microstrain"])), flush=True)


if __name__ == "__main__":
    main()
