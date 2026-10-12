"""Assess regional history information and query a changed control protocol."""

import argparse
import json

import numpy as np
import torch

from analyze_matched_history_response import regional_contrast
from train_control_history import RESULT, SOURCE, load_pilot, responses, summary


def compare_history(values, reference_host, reference_circuit):
    pair = [30, 32]
    actual = regional_contrast(values["host_microstrain"][pair].reshape(2, 2, 3, 2),
                               values["circuit_microstrain"][pair].reshape(2, 2, 3, 2))
    reference = regional_contrast(reference_host[pair].reshape(2, 2, 3, 2),
                                  reference_circuit[pair].reshape(2, 2, 3, 2))
    difference = {name: np.asarray(actual[f"{name}_microstrain"])-reference[f"{name}_microstrain"]
                  for name in ("host", "circuit")}
    difference["discrepancy"] = difference["circuit"]-difference["host"]
    error = {name: float(np.sqrt(np.mean(value**2))) for name, value in difference.items()}
    return actual, reference, error


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(RESULT))
    parser.add_argument("--query-reference", help="Saved local-physics evaluation of the 0.7-mm reversal")
    args = parser.parse_args()
    root = SOURCE.parents[2]
    checkpoint = root/args.checkpoint
    problem, network, eta, prefix, prefix_forces, controls, fields, forces, readout, _ = load_pilot()
    network.load_state_dict(torch.load(checkpoint.with_suffix(".pt"), weights_only=True)["network"])
    saved = dict(np.load(checkpoint.with_suffix(".npz")))
    q = ((fields-fields[:1])@problem.base.functionals.q_matrix.T).numpy()*1e6
    circuit = readout(fields@problem.base.raw_matrix.T).flatten(1)
    circuit = (circuit-circuit[:1]).numpy()*1e6
    network_contrast, source_contrast, errors = compare_history(saved, q, circuit)
    np.testing.assert_array_equal(saved["displacement_m"][:len(prefix[0])], prefix[0])
    short_fields, short_forces = network(eta, prefix, prefix_forces, controls[:, :2])
    np.testing.assert_allclose(short_fields[0], saved["displacement_m"][:32], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(short_forces[0], saved["force_n"][:32], rtol=1e-9, atol=1e-5)
    # The query uses its actual controls through the same learned weights.
    query_controls = controls.new_tensor([[.6, .7, .6]])*1e-3
    query = responses(problem, network, eta, prefix, prefix_forces, query_controls, readout)
    report = dict(checkpoint=checkpoint.relative_to(root).as_posix(),
        source_states=SOURCE.name, role="Shared nominal-material control-history pilot",
        fixed_prefix_preserved=True, shorter_query_preserves_shared_prefix=True,
        saved_physical_history=source_contrast, trained_history=network_contrast,
        history_contrast_error_rms_microstrain=errors,
        training_summary=summary(saved),
        changed_protocol=dict(controls_mm=(query_controls[0]*1e3).tolist(),
            role="Untrained lower-peak query; its fields have not received a separate physical correction",
            **summary(query)))
    if args.query_reference:
        reference_path = root/args.query_reference
        reference_report = json.loads(reference_path.read_text())
        if reference_report["termination"] != "local_targets_completed":
            raise ValueError("The query reference must contain a completed local protocol")
        physical = np.load(reference_path.with_suffix(".npz"))
        query_fields = problem.physics.tensor(physical["displacement_m"])
        np.testing.assert_allclose(((query_fields[30:]-query_fields[:1])@network.balance.local)*1e3,
                                   [.6, .7, .6], rtol=0, atol=1e-8)
        q_query = ((query_fields-query_fields[:1])@problem.base.functionals.q_matrix.T).numpy()*1e6
        c_query = readout(query_fields@problem.base.raw_matrix.T).flatten(1)
        c_query = (c_query-c_query[:1]).numpy()*1e6
        actual, reference, error = compare_history(query, q_query, c_query)
        report["changed_protocol"].update(
            role="Untrained lower-peak query assessed against local corrections in the same physical field",
            physical_evaluation=reference_path.relative_to(root).as_posix(),
            physical_tail_force_kn=(physical["force_n"][30:]/1e3).tolist(),
            predicted_history=actual, physical_history=reference,
            history_contrast_error_rms_microstrain=error)
    training_record = json.loads(checkpoint.read_text())
    initialization = training_record.get("initialization_checkpoint")
    if initialization and all(row["method"] in ("gauss_newton", "regional_history")
                              for row in training_record["iterations"]):
        # The encoder is fixed in these shared-head updates. Determine how a
        # change learned at the training controls reaches the actual new query.
        feature = network.features(eta, prefix, controls)[0]
        query_feature = network.features(eta, prefix, query_controls)[0]
        feature = torch.cat([feature, torch.ones_like(feature[:, :1])], dim=-1).numpy()
        query_feature = torch.cat([query_feature, torch.ones_like(query_feature[:, :1])], dim=-1).numpy()
        transfer = query_feature@np.linalg.pinv(feature)
        initial = torch.load((root/initialization).with_suffix(".pt"), weights_only=True)["network"]
        for name, value in network.encoder.state_dict().items():
            torch.testing.assert_close(value, initial["encoder."+name], rtol=0, atol=0)
        update = np.vstack([(network.head.weight-initial["head.weight"]).numpy().T,
                            (network.head.bias-initial["head.bias"]).numpy()])
        np.testing.assert_allclose(query_feature@update, transfer@(feature@update), rtol=1e-7, atol=1e-10)
        report["changed_protocol"]["shared_head_update_transfer"] = dict(
            query_state_weights=transfer.tolist(),
            history_contrast_weights=(transfer[2]-transfer[0]).tolist(),
            training_controls_mm=(controls[0]*1e3).tolist(),
            role="Exact transfer of this fixed-encoder shared-head update; coefficients map training raw-field changes to query changes, not complete physical predictions")
    target = checkpoint.with_name("B2_control_history_response.json" if checkpoint == RESULT
                                   else checkpoint.stem+"_response.json")
    target.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report["history_contrast_error_rms_microstrain"]), flush=True)


if __name__ == "__main__":
    main()
