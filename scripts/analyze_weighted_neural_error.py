"""Neural residual effects on finite passive-shunt design decisions.

The full-history local correction and its refreshed remainder concern the
saved PINN paths on each fixed P2 representation. They are diagnostic states,
not independent reference solutions or an error bound for the continuum.
"""

import argparse
import gc
import json
import time

import numpy as np
import torch

from analyze_spatial_transfer import raw_gauges
from analyze_weighted_readout import OUTPUT, SOURCE, fit, rms
from functional_readout import PathLinearization, ReadoutFunctionals
from sparse_elastic_coordinates import SpatialCoordinates
from spatial_resolution import load_resolution_problem
from stateful_physics import ROOT
from weighted_readout import shunt_design, uniform_parameters, weight_map, weighted_response


def objective_vector(difference):
    """The declared absolute and finite-field objective, in consistent units."""
    change = difference[1:]-difference[:1]
    return np.r_[difference.ravel()/np.sqrt(difference.size),
                 change.ravel()/np.sqrt(change.size)]


def circuit_metrics(data, parameters, settings):
    result = weighted_response(data["raw"], parameters, settings)
    difference = (result["readout"]-data["target"])*1e6
    vector = objective_vector(difference)
    return dict(mismatch_rms_microstrain=rms(difference),
        mismatch_change_rms_microstrain=rms(difference[1:]-difference[:1]),
        objective_microstrain2=float(vector@vector),
        objective_root_microstrain=float(np.linalg.norm(vector))), difference


def state_objective_correction(data, delta, parameters, settings):
    """Differentiate the state, holding installed resistors/calibration fixed."""
    value = weighted_response(data["raw"], parameters, settings)["readout"]
    differential = weighted_response(data["raw"]+1e-25j*delta["raw"],
                                     parameters, settings)["readout"].imag/1e-25
    mismatch = (value-data["target"])*1e6
    delta_mismatch = (differential-delta["target"])*1e6
    return float(2*objective_vector(mismatch)@objective_vector(delta_mismatch))


def collect_corrections(name, device):
    report = json.loads((SOURCE/f"{name}.json").read_text())
    reference = json.loads((SOURCE/f"{report['reference']}.json").read_text())
    with np.load(SOURCE/f"{name}.npz") as saved:
        data = dict(saved)
    p = load_resolution_problem(reference["mesh_size_m"], reference["order"],
                                device, reference["local_size_m"])
    all_coordinates = SpatialCoordinates(p, data["bond_scale"])
    functionals = ReadoutFunctionals(p)
    target_matrix = functionals.q_matrix.detach().cpu().numpy()
    raw_matrix = raw_gauges(p)[0][0]
    free = p.free.cpu().numpy()
    raw_corrections, target_corrections, diagnostics = [], [], []
    for i, label in enumerate(data["case_labels"]):
        started = time.perf_counter()
        u = p.tensor(data["displacement_m"][i])
        scale = p.tensor(data["bond_scale"][i:i+1])
        coordinates = all_coordinates.for_field(data["bond_scale"][i:i+1])
        projection = coordinates.projection
        force = p.forces(u[None], bond_scale=scale)[0][0, :, p.free].detach().cpu().numpy()
        # Detect source/model drift before using the saved residual as evidence.
        np.testing.assert_allclose(force, data["free_force_n"][i], atol=5e-8, rtol=1e-7)
        np.testing.assert_allclose(u.cpu().numpy()@target_matrix.T,
            data["host_target"][i].reshape(5, 12), atol=1e-16, rtol=1e-10)
        np.testing.assert_allclose(u.cpu().numpy()@raw_matrix.T,
            data["raw_gauge_strain"][i, :, 0], atol=1e-16, rtol=1e-10)
        stages, draw, dtarget = [], [], []
        # The second linearization tests the unaccounted local remainder and
        # re-evaluates contact/cohesive branches at the diagnostic state.
        for stage in range(2):
            linear = PathLinearization(p, u, bond_scale=scale,
                                       projection=projection, factorize=False)
            residual = np.stack([projection.T@value for value in force])
            reduced = coordinates.correct_path(linear, -residual)
            lifted = np.stack([projection@value for value in reduced])
            dq = lifted@target_matrix[:, free].T
            de = lifted@raw_matrix[:, free].T
            draw.append(de)
            dtarget.append(dq.reshape(5, 2, 3, 2))
            rejected = force-np.stack([projection@value for value in residual])
            stage_record = dict(stage=stage+1,
                target_correction_rms_microstrain=rms(dq)*1e6,
                gauge_correction_rms_microstrain=rms(de)*1e6,
                rejected_force_rms_n=rms(rejected),
                linear_equation_relative_residual=float(
                    np.linalg.norm(linear.apply(reduced)+residual)/np.linalg.norm(residual)))
            du = torch.zeros_like(u)
            du[:, p.free] = p.tensor(lifted)
            u = u+du
            next_force = p.forces(u[None], bond_scale=scale)[0][0, :, p.free].detach().cpu().numpy()
            stage_record["nonlinear_residual_norm_ratio"] = float(
                np.linalg.norm(next_force)/np.linalg.norm(force))
            stages.append(stage_record)
            force = next_force
            del linear
            gc.collect()
        row = dict(case=str(label), stages=stages,
                   elapsed_seconds=time.perf_counter()-started)
        diagnostics.append(row)
        raw_corrections.append(draw)
        target_corrections.append(dtarget)
        print(json.dumps(dict(source=name, **row)), flush=True)
    result = dict(raw_correction=np.stack(raw_corrections),
                  target_correction=np.stack(target_corrections),
                  case_labels=data["case_labels"])
    np.savez_compressed(OUTPUT/f"{name}_neural_corrections.npz", **result)
    (OUTPUT/f"{name}_neural_corrections.json").write_text(json.dumps(dict(
        source=name, reference=report["reference"], cases=diagnostics,
        interpretation="Two sequential full-history residual linearizations on the same P2 physics. The first estimates the saved neural state error; the refreshed second tests the remaining local effect. No diagnostic state is used as a training label or independent mechanical truth."),
        indent=2, allow_nan=False)+"\n")
    del p, all_coordinates, data
    gc.collect()


def design_values(parameters, settings):
    weights, rho, _ = shunt_design(parameters, settings)
    mask = weight_map().sum(axis=-1) > 0
    resistance = [[None if r == 1 else float(settings["sensor_resistance_ohm"]*r/(1-r))
                   for r in row[present]] for row, present in zip(rho, mask)]
    return dict(parameters=parameters.tolist(),
                weights=[row[present].tolist() for row, present in zip(weights, mask)],
                retention=[row[present].tolist() for row, present in zip(rho, mask)],
                shunt_resistance_ohm=resistance)


def analyze(names):
    setup = json.loads((ROOT/"inputs/weighted_readout.json").read_text())
    settings = json.loads((ROOT/setup["settings_source"]).read_text())
    previous = json.loads((OUTPUT/"comparison.json").read_text())["designs"]
    records, saved_arrays = {}, {}
    for name in names:
        with np.load(SOURCE/f"{name}.npz") as saved:
            raw, target = saved["raw_gauge_strain"][:, :, 0], saved["host_target"]
            labels = saved["case_labels"]
        with np.load(OUTPUT/f"{name}_neural_corrections.npz") as saved:
            np.testing.assert_array_equal(labels, saved["case_labels"])
            dr, dq = saved["raw_correction"], saved["target_correction"]
        datasets = {"saved": dict(raw=raw, target=target),
                    "first_correction": dict(raw=raw+dr[:, 0], target=target+dq[:, 0]),
                    "refreshed_correction": dict(raw=raw+dr.sum(axis=1), target=target+dq.sum(axis=1))}
        designs = {"uniform": uniform_parameters()}
        fits = {}
        for label, data in datasets.items():
            designs[label], fits[label] = fit(data, list(range(len(labels))))
        np.testing.assert_allclose(designs["saved"], previous[f"{name}_all_fields"]["parameters"],
                                   atol=1e-10, rtol=1e-9)
        design_records = {}
        for design, parameters in designs.items():
            evaluations = {}
            differences = {}
            for label, data in datasets.items():
                evaluations[label], differences[label] = circuit_metrics(data, parameters, settings)
                saved_arrays[f"{name}__{design}__{label}__mismatch_microstrain"] = differences[label]
            correction = state_objective_correction(datasets["saved"],
                dict(raw=dr[:, 0], target=dq[:, 0]), parameters, settings)
            evaluations["linearized_objective_correction_microstrain2"] = correction
            evaluations["finite_minus_linearized_objective_correction_microstrain2"] = (
                evaluations["first_correction"]["objective_microstrain2"]-
                evaluations["saved"]["objective_microstrain2"]-correction)
            evaluations["first_mismatch_correction_rms_microstrain"] = rms(
                differences["first_correction"]-differences["saved"])
            evaluations["refreshed_mismatch_correction_rms_microstrain"] = rms(
                differences["refreshed_correction"]-differences["first_correction"])
            design_records[design] = dict(**design_values(parameters, settings), evaluations=evaluations)
        finite_changes = []
        for a, b in [("uniform", "saved"), ("saved", "first_correction"),
                     ("first_correction", "refreshed_correction"),
                     ("saved", "refreshed_correction")]:
            changes = {label: design_records[b]["evaluations"][label]["objective_microstrain2"]-
                       design_records[a]["evaluations"][label]["objective_microstrain2"]
                       for label in datasets}
            finite_changes.append(dict(from_design=a, to_design=b,
                objective_change_microstrain2=changes,
                first_neural_correction_to_design_change_microstrain2=(
                    changes["first_correction"]-changes["saved"]),
                linearized_neural_correction_to_design_change_microstrain2=(
                    design_records[b]["evaluations"]["linearized_objective_correction_microstrain2"]-
                    design_records[a]["evaluations"]["linearized_objective_correction_microstrain2"]),
                refreshed_remainder_to_design_change_microstrain2=(
                    changes["refreshed_correction"]-changes["first_correction"]),
                max_parameter_change=float(np.max(abs(designs[b]-designs[a]))),
                max_retention_change=float(np.max(abs(
                    shunt_design(designs[b], settings)[1]-shunt_design(designs[a], settings)[1]))),
                readout_change_on_refreshed_state_rms_microstrain=rms(
                    saved_arrays[f"{name}__{b}__refreshed_correction__mismatch_microstrain"]-
                    saved_arrays[f"{name}__{a}__refreshed_correction__mismatch_microstrain"])))
        records[name] = dict(fits=fits, designs=design_records, finite_design_changes=finite_changes,
            target_first_correction_rms_microstrain=rms(dq[:, 0])*1e6,
            target_refreshed_correction_rms_microstrain=rms(dq[:, 1])*1e6)
        print(json.dumps(dict(source=name, finite_design_changes=finite_changes)), flush=True)
    result = dict(sources=records,
        interpretation="Finite resistor designs are selected with the original linear synthesis and evaluated with the exact loaded circuit and their fixed zero/gain calibration. Differences between saved and residual-corrected data estimate neural error on the same P2 representation. The refreshed correction diagnoses the local remainder; it is not an independent reference, continuum-error bound or a trained spatial surrogate test.")
    (OUTPUT/"neural_error_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(OUTPUT/"neural_error_responses.npz", **saved_arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=["finite_p2_local030", "finite_p2_local020"])
    parser.add_argument("--reuse-corrections", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not args.reuse_corrections:
        for name in args.names:
            collect_corrections(name, args.device)
    analyze(args.names)


if __name__ == "__main__":
    main()
