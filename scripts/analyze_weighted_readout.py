"""Synthesize passive weights from saved neural paths and cross-evaluate them."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from weighted_readout import (PARAMETER_GROUPS, linear_features,
                             uniform_parameters, weight_map, weighted_response)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"results/ideation/stateful_pilot/spatial_resolution"
OUTPUT = ROOT/"results/ideation/stateful_pilot/weighted_readout"


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def load_paths(name):
    record = json.loads((SOURCE/f"{name}.json").read_text())
    with np.load(SOURCE/f"{record['reference']}_local.npz") as tangent:
        raw_tangent = tangent["raw_gauge_tangent_microstrain"][0]*1e-6
        target_tangent = tangent["target_tangent_microstrain"]
    with np.load(SOURCE/f"{name}.npz") as saved:
        return dict(raw=saved["raw_gauge_strain"][:, :, 0],
                    target=saved["host_target"], labels=saved["case_labels"],
                    old_readout=saved["circuit_equivalent_strain"][:, :, 0],
                    raw_tangent=raw_tangent, target_tangent=target_tangent)


def objective_system(data, selection):
    features = linear_features(data["raw"])*1e6
    target = data["target"]*1e6
    x = features[selection].reshape(-1, 16)
    y = target[selection].ravel()
    parts_x, parts_y = [x/np.sqrt(len(y))], [y/np.sqrt(len(y))]
    changing = [i for i in selection if i != 0]
    if changing:
        dx = (features[changing]-features[:1]).reshape(-1, 16)
        dy = (target[changing]-target[:1]).ravel()
        parts_x.append(dx/np.sqrt(len(dy)))
        parts_y.append(dy/np.sqrt(len(dy)))
    return np.concatenate(parts_x), np.concatenate(parts_y)


def fit(data, selection):
    x, y = objective_system(data, selection)
    constraint = np.zeros((4, 16))
    for row, group in enumerate(PARAMETER_GROUPS):
        constraint[row, group] = 1.

    def objective(a):
        error = x@a-y
        return error@error, 2*x.T@error

    result = minimize(objective, uniform_parameters(), jac=True, method="SLSQP",
        constraints=LinearConstraint(constraint, np.ones(4), np.ones(4)),
        bounds=Bounds(np.zeros(16), np.ones(16)), options=dict(ftol=1e-11, maxiter=500))
    if not result.success:
        raise RuntimeError(result.message)
    # Constraint tangent identifies poorly determined physical reweightings.
    _, _, vh = np.linalg.svd(constraint, full_matrices=True)
    singular = np.linalg.svd(x@vh[4:].T, compute_uv=False)
    return result.x, dict(message=result.message, iterations=int(result.nit),
        linear_objective_microstrain2=float(result.fun),
        design_tangent_singular_values=singular.tolist())


def evaluate(data, parameters, settings):
    circuit = weighted_response(data["raw"], parameters, settings)
    difference = (circuit["readout"]-data["target"])*1e6
    change = difference-difference[:1]
    linear = np.einsum("...p,p->...", linear_features(data["raw"]), parameters)
    gain = circuit["gain"]
    readout_tangent = np.stack([weighted_response(
        data["raw"][0]+1e-25j*data["raw_tangent"][..., j], parameters, settings
        )["readout"].imag.reshape(5, 12)/1e-25*1e6
        for j in range(data["raw_tangent"].shape[-1])], axis=-1)
    tangent_difference = readout_tangent-data["target_tangent"]
    rows = []
    for i, label in enumerate(data["labels"]):
        rows.append(dict(case=str(label), mismatch_rms_microstrain=rms(difference[i]),
            mismatch_change_rms_microstrain=rms(change[i])))
    record = dict(cases=rows, mismatch_rms_microstrain=rms(difference),
        mismatch_change_rms_microstrain=rms(change[1:]),
        objective_root_microstrain=float(np.sqrt(np.mean(difference**2)+np.mean(change[1:]**2))),
        exact_minus_linear_rms_microstrain=rms((circuit["readout"]-linear)*1e6),
        full_local_tangent_mismatch_rms_microstrain=rms(tangent_difference),
        local_tangent_mismatch_by_parameter_microstrain=np.sqrt(
            np.mean(tangent_difference**2, axis=(0, 1))).tolist(),
        completion_resistance_ohm=circuit["completion"].tolist(),
        gain_v_per_strain=gain.tolist(),
        bridge_input_gain_v_per_strain=(gain/settings["amplifier_gain"]).tolist(),
        twelve_bridge_power_w_range=[float(circuit["power"].sum(axis=-1).min()),
                                    float(circuit["power"].sum(axis=-1).max())],
        amplifier_voltage_range_v=[float(circuit["voltage"].min()), float(circuit["voltage"].max())],
        gauge_strain_abs_max_microstrain=float(np.abs(data["raw"]).max()*1e6))
    return record, difference, circuit


def main():
    setup = json.loads((ROOT/"inputs/weighted_readout.json").read_text())
    settings = json.loads((ROOT/setup["settings_source"]).read_text())
    names = setup["mechanical_sources"]
    datasets = {name: load_paths(name) for name in names}
    np.testing.assert_array_equal(datasets[names[0]]["labels"], datasets[names[1]]["labels"])
    labels = list(datasets[names[0]]["labels"])
    selections = {"all_fields": list(range(len(labels)))}
    shapes = json.loads((ROOT/"inputs/spatial_resolution.json").read_text())["spatial_followup"]["finite_shapes"]
    for shape in shapes:
        selections[f"without_{shape}"] = [i for i, label in enumerate(labels) if not str(label).startswith(shape+"_")]
    records, arrays = {}, {}
    designs = [("uniform", uniform_parameters(), None)]
    for source in names:
        for subset, selection in selections.items():
            parameters, fit_record = fit(datasets[source], selection)
            fit_record.update(source=source, training_cases=[str(labels[i]) for i in selection])
            designs.append((f"{source}_{subset}", parameters, fit_record))
    for name, parameters, fit_record in designs:
        row = dict(parameters=parameters.tolist(), fit=fit_record, evaluations={})
        for destination, data in datasets.items():
            metrics, difference, circuit = evaluate(data, parameters, settings)
            if name == "uniform":
                np.testing.assert_allclose(circuit["readout"], data["old_readout"], atol=2e-15, rtol=1e-11)
            row["evaluations"][destination] = metrics
            if fit_record is not None:
                held = [i for i, label in enumerate(data["labels"])
                        if label not in fit_record["training_cases"]]
                if held:
                    metrics["held_shapes"] = dict(cases=[str(data["labels"][i]) for i in held],
                        mismatch_rms_microstrain=rms(difference[held]),
                        mismatch_change_rms_microstrain=rms(difference[held]-difference[:1]))
            arrays[f"{name}__{destination}__mismatch_microstrain"] = difference
        # Store only physically present connections; absent cross-group gauges
        # are not bypass resistors. null denotes an open shunt.
        mask = weight_map().sum(axis=-1) > 0
        row["individual_weights"] = [w[present].tolist() for w, present in zip(circuit["weights"], mask)]
        r0 = settings["sensor_resistance_ohm"]
        row["shunt_resistance_ohm"] = [[None if r == 1 else float(r0*r/(1-r))
                                       for r in rho[present]]
                                      for rho, present in zip(circuit["rho"], mask)]
        records[name] = row
    result = dict(settings=setup, designs=records,
        interpretation="The fitted objective is a convex positive-weight response synthesis on saved neural fields. Evaluation uses exact shunt resistances and loaded bridges with a fixed uniform-strain calibration. Cross-mesh and held-shape outcomes concern this design family; they are not PINN training-method comparisons.")
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(OUTPUT/"responses.npz", **arrays)
    for name, row in records.items():
        print(json.dumps(dict(design=name, objective={k:v["objective_root_microstrain"]
            for k,v in row["evaluations"].items()})), flush=True)


if __name__ == "__main__":
    main()
