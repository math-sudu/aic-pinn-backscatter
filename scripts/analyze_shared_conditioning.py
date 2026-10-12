"""Evaluate shared training against the saved actual-query diagnosis."""

import argparse
import json

import numpy as np

from analyze_weighted_neural_error import circuit_metrics, design_values
from analyze_weighted_readout import rms
from shared_spatial_surrogate import OUTPUT, ROOT
from weighted_readout import weighted_response


def nominal_change_components(errors, nominal_error):
    """Split nominal-subtracted MSE into condition spread and reference offset."""
    mean_error = errors.mean(axis=0)
    return dict(centered_rms_microstrain=rms(errors-mean_error),
                nominal_offset_rms_microstrain=rms(mean_error-nominal_error))


def response_errors(data, reference, parameters, settings):
    target = (data["target"]-reference["target"])*1e6
    readout = (weighted_response(data["raw"], parameters, settings)["readout"]-
               weighted_response(reference["raw"], parameters, settings)["readout"])*1e6
    mismatch = readout-target
    reflected_target = target.copy()
    reflected_target[:, :, 0] = target[:, :, 0, ::-1]
    return dict(host_rms_microstrain=rms(target), readout_rms_microstrain=rms(readout),
        finite_host_change_rms_microstrain=rms(target[1:]-target[:1]),
        finite_readout_change_rms_microstrain=rms(readout[1:]-readout[:1]),
        mismatch_rms_microstrain=rms(mismatch),
        finite_mismatch_change_rms_microstrain=rms(mismatch[1:]-mismatch[:1]),
        nominal_mismatch_rms_microstrain=rms(mismatch[0]),
        mixed_mismatch_rms_microstrain=rms(mismatch[7:]),
        mixed_finite_mismatch_change_rms_microstrain=rms(mismatch[7:]-mismatch[:1]),
        finite_mismatch_components=dict(
            all_nonnominal=nominal_change_components(mismatch[1:], mismatch[0]),
            mixed=nominal_change_components(mismatch[7:], mismatch[0])),
        endpoint_host_rms_microstrain=rms(target[:7]),
        mixed_host_rms_microstrain=rms(target[7:]),
        even_host_rms_microstrain=rms((target+reflected_target)/2),
        odd_host_rms_microstrain=rms((target-reflected_target)/2),
        path_host_rms_microstrain=[rms(target[:, i]) for i in range(target.shape[1])])


def analyze(name, reference_name):
    record = json.loads((OUTPUT/f"{name}.json").read_text())
    reference_record = json.loads((OUTPUT/f"{reference_name}_comparison.json").read_text())
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    datasets = {}
    for role, stem in (("baseline", reference_name), ("conditioned", name)):
        with np.load(OUTPUT/f"{stem}_query.npz") as saved:
            datasets[role] = dict(raw=saved["raw_gauge_strain"], target=saved["host_target"])
            if role == "baseline":
                labels, eta = saved["case_labels"], saved["parameters"]
            else:
                np.testing.assert_array_equal(saved["case_labels"], labels)
                np.testing.assert_array_equal(saved["parameters"], eta)
    with np.load(OUTPUT/f"{reference_name}_corrections.npz") as saved:
        np.testing.assert_array_equal(saved["case_labels"], labels)
        np.testing.assert_array_equal(saved["source_parameters"], eta)
        np.testing.assert_array_equal(saved["source_target"], datasets["baseline"]["target"])
        np.testing.assert_array_equal(saved["source_gauges"], datasets["baseline"]["raw"])
        reference = dict(raw=saved["source_gauges"]+saved["raw_correction"].sum(axis=1),
                         target=saved["source_target"]+saved["target_correction"].sum(axis=1))
    designs = dict(baseline=np.asarray(reference_record["designs"]["shared"]["parameters"]),
        conditioned=np.asarray(record["training_history"][-1]["parameters"]),
        corrected=np.asarray(reference_record["designs"]["refreshed_correction"]["parameters"]))
    records = {}
    for design, parameters in designs.items():
        records[design] = dict(**design_values(parameters, settings),
            response_errors={role: response_errors(data, reference, parameters, settings)
                             for role, data in datasets.items()},
            evaluations={role: circuit_metrics(data, parameters, settings)[0]
                         for role, data in dict(datasets, corrected=reference).items()})
    changes = []
    for before, after in (("baseline", "conditioned"), ("conditioned", "corrected")):
        changes.append(dict(from_design=before, to_design=after,
            readout_change_on_corrected_state_rms_microstrain=rms(
                weighted_response(reference["raw"], designs[after], settings)["readout"]-
                weighted_response(reference["raw"], designs[before], settings)["readout"])*1e6,
            objective_root_change_microstrain=records[after]["evaluations"]["corrected"][
                "objective_root_microstrain"]-records[before]["evaluations"]["corrected"][
                "objective_root_microstrain"],
            max_parameter_change=float(np.max(abs(designs[after]-designs[before])))))
    result = dict(name=name, reference=reference_name, training_cycles=len(record["training_history"]),
        reference_finite_host_change_rms_microstrain=rms(
            reference["target"][1:]-reference["target"][:1])*1e6,
        designs=records, design_changes=changes,
        cases=[dict(case=str(label), baseline_host_rms_microstrain=rms(
            datasets["baseline"]["target"][i]-reference["target"][i])*1e6,
            conditioned_host_rms_microstrain=rms(
            datasets["conditioned"]["target"][i]-reference["target"][i])*1e6)
            for i, label in enumerate(labels)],
        interpretation="The original 23 query conditions and cached two-stage full-history diagnostic states are unchanged. Those states are used only for evaluation, never for training. Fixed-circuit host/readout errors and finite changes are reported separately. The named checkpoint's setup and initialization specify the training intervention; this comparison measures its response and circuit effects on the common query set.")
    (OUTPUT/f"{name}_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(name=name, fixed_circuit=records["baseline"]["response_errors"],
                          design_changes=changes)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="physics_conditioned")
    parser.add_argument("--reference", default="physics")
    args = parser.parse_args()
    analyze(args.name, args.reference)
