"""Compare neural interventions on one common local physics correction.

Four material corners and the four held materials cover the existing domain.
The reference is one complete-history Newton correction of the shared starting
PINN, not an experimental target or an independently converged solution.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import time

import numpy as np
import torch

from analyze_fixed_reference_history import fixed_circuit, metrics
from analyze_shared_conditioning import nominal_change_components
from fixed_reference_history import (ROOT, OUTPUT, FixedReferenceNetwork, MaterialHistoryProblem,
                                     absolute_components, parameter_indices)
from history_optimization import initialize_worker, worker_direction


def prepare_reference(source, workers):
    record = json.loads((OUTPUT/f"{source}.json").read_text())
    setup = record["setup"]
    parameters = np.asarray([[1., -1.], [1., 1.], [-1., -1.], [-1., 1.]]+setup["query_parameters"])
    problem = MaterialHistoryProblem(setup)
    p = problem.physics
    network = FixedReferenceNetwork(problem.coordinates, problem.initial, setup["width"], setup["path_coordinates"])
    network.load_state_dict(torch.load(OUTPUT/f"{source}.pt", weights_only=True)["network"])
    with torch.no_grad():
        raw = network.raw(p.tensor(parameters)).numpy()
        states = network(p.tensor(parameters)).numpy()
    payloads = [(state, eta, network.path_coordinates) for state, eta in zip(states, parameters)]
    host, gauges, diagnostics = [], [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker, initargs=(setup,)) as executor:
        for i, (encoded, info) in enumerate(executor.map(worker_direction, payloads)):
            corrected_raw = p.tensor(raw[i:i+1]+encoded)
            absolute = (absolute_components(corrected_raw) if network.path_coordinates == "reference_increments"
                        else corrected_raw)
            with torch.no_grad():
                state = problem.coordinates.decode(absolute)*p.scale
                force, damage, history = problem.equilibrium(state, p.tensor(parameters[i:i+1]))
                before = problem.residual(network, p.tensor(parameters[i:i+1])).square().mean()
                after = problem.coordinates.whiten(force).square().mean()
                increments = state-state[:, :1]
                host.append((increments@problem.functionals.q_matrix.T).reshape(network.shape[0], 2, 3, 2).numpy())
                gauges.append((state@problem.raw_matrix.T)[0].numpy())
            diagnostics.append(dict(parameters=parameters[i].tolist(),
                physics_before=float(before), physics_after=float(after),
                maximum_damage=float(damage.max()), peak_history_micro_m=float(history.max())*1e6,
                **info))
            print(json.dumps(dict(reference_case=i+1, total=len(parameters),
                                  seconds=time.perf_counter()-started, **diagnostics[-1])), flush=True)
    data = dict(parameters=parameters, force_n=problem.force[0].numpy(),
                host_increment=np.stack(host), raw_gauge_strain=np.stack(gauges))
    np.savez_compressed(OUTPUT/"regional_history_reference.npz", **data)
    report = dict(source=source, material_parameters=parameters.tolist(),
        corner_indices=list(range(4)), held_indices=list(range(4, len(parameters))), nominal_index=0,
        correction_diagnostics=diagnostics, seconds=time.perf_counter()-started,
        definition="One full-history Newton correction of the starting shared PINN at four material corners and four held inputs. Exact nonlinear circuits use the corrected raw gauges. All methods and selected circuits are assessed on these same fields. The four held inputs enter neither neural training nor resistor selection. These are local physics diagnostics, not experimental truth or a converged field reference.")
    (OUTPUT/"regional_history_reference.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


def response_distances(predicted, reference, installed, settings, indices, nominal):
    predicted_circuit, _ = fixed_circuit(predicted, installed, settings)
    reference_circuit, _ = fixed_circuit(reference, installed, settings)
    host = (predicted["host_increment"]-reference["host_increment"])*1e6
    circuit = (predicted_circuit-reference_circuit)*1e6
    errors = dict(host=host[:, 1:], fixed_circuit=circuit[:, 1:], discrepancy=(circuit-host)[:, 1:])
    result = {}
    for block, values in errors.items():
        change = values[indices]-values[nominal]
        changed = np.asarray(indices) != nominal
        decomposition = nominal_change_components(values[np.asarray(indices)[changed]], values[nominal])
        change_mse = np.mean(change[changed]**2)
        np.testing.assert_allclose(sum(value**2 for value in decomposition.values()), change_mse, rtol=1e-12)
        complete = np.concatenate([np.zeros_like(values[:, :1]), values], axis=1)
        result[block] = dict(value_rms_microstrain=float(np.sqrt(np.mean(values[indices]**2))),
            material_change_rms_microstrain=float(np.sqrt(change_mse)),
            material_change_components=decomposition,
            loading_increment_rms_microstrain=float(np.sqrt(np.mean(np.diff(complete[indices], axis=1)**2))),
            by_step_rms_microstrain=np.sqrt(np.mean(values[indices]**2, axis=(0, 2, 3, 4))).tolist(),
            by_direction_region_component_rms_microstrain=np.sqrt(np.mean(values[indices]**2, axis=(0, 1))).tolist())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="physics_history_metric")
    parser.add_argument("--names", nargs="+", default=["physics_history_metric", "physics_regional_control", "regional_history"])
    parser.add_argument("--prepare-reference", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.prepare_reference:
        prepare_reference(args.source, args.workers)
        return
    reference = dict(np.load(OUTPUT/"regional_history_reference.npz"))
    reference_record = json.loads((OUTPUT/"regional_history_reference.json").read_text())
    if reference_record["source"] != args.source:
        raise ValueError("The reference belongs to another starting checkpoint")
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    design_record = json.loads((OUTPUT/f"{args.source}_readout.json").read_text())
    installed = design_record["designs"]["selected"]["parameters"]
    nominal = reference_record["nominal_index"]
    groups = dict(corners=reference_record["corner_indices"], held=reference_record["held_indices"])
    models = {}
    for name in args.names:
        saved = dict(np.load(OUTPUT/f"{name}_query.npz"))
        indices = parameter_indices(saved["parameters"], reference["parameters"])
        predicted = {key: value if key == "force_n" else value[indices] for key, value in saved.items()}
        np.testing.assert_array_equal(predicted["parameters"], reference["parameters"])
        np.testing.assert_array_equal(predicted["force_n"], reference["force_n"])
        readout = json.loads((OUTPUT/f"{name}_readout.json").read_text())
        selected = readout["designs"]["selected"]["parameters"]
        models[name] = dict(
            response_distance={group: response_distances(predicted, reference, installed, settings, rows, nominal)
                               for group, rows in groups.items()},
            selected_design_on_common_fields={group: metrics(reference, selected, settings, rows, nominal)
                                              for group, rows in groups.items()},
            selected_design_parameters=selected)
    result = dict(source=args.source, reference="regional_history_reference.npz",
        reference_definition=reference_record["definition"], material_parameters=reference["parameters"].tolist(),
        force_kn=(reference["force_n"]/1000).tolist(), groups=groups, installed_design=installed,
        models=models)
    (OUTPUT/"regional_history_comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({name: dict(held_response={key: {k: v for k, v in value.items() if not k.startswith("by_")}
        for key, value in row["response_distance"]["held"].items()},
        held_design=row["selected_design_on_common_fields"]["held"]) for name, row in models.items()}), flush=True)


if __name__ == "__main__":
    main()
