"""Evaluate fixed-reference regional outputs and one installed shunt design."""

import argparse
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
import torch

from analyze_weighted_neural_error import design_values
from fixed_reference_history import (ROOT, OUTPUT, MaterialHistoryProblem, FixedReferenceNetwork,
                                     absolute_components, parameter_indices)
from weighted_readout import PARAMETER_GROUPS, FixedShuntReadout, uniform_parameters, weighted_response


def fixed_circuit(data, parameters, settings):
    circuit = weighted_response(data["raw_gauge_strain"], parameters, settings)
    values = circuit["readout"]
    return values-values[:, :1], circuit


def metrics(data, parameters, settings, selection, nominal):
    values, circuit = fixed_circuit(data, parameters, settings)
    mismatch = (values-data["host_increment"])[:, 1:]*1e6
    change = mismatch[selection]-mismatch[nominal]
    changed = np.asarray(selection) != nominal
    value_mse = np.mean(mismatch[selection]**2)
    change_mse = np.mean(change[changed]**2)
    return dict(response_rms_microstrain=float(np.sqrt(value_mse)),
        material_change_rms_microstrain=float(np.sqrt(change_mse)),
        objective_microstrain2=float(value_mse+change_mse),
        objective_root_microstrain=float(np.sqrt(value_mse+change_mse)))


def select_design(data, settings, selection, nominal):
    constraint = np.zeros((len(PARAMETER_GROUPS), 16))
    for i, group in enumerate(PARAMETER_GROUPS):
        constraint[i, group] = 1.

    def objective(parameters):
        physical = parameters.copy()
        for group in PARAMETER_GROUPS:
            physical[group] /= physical[group].sum()
        return metrics(data, physical, settings, selection, nominal)["objective_microstrain2"]

    result = minimize(objective, uniform_parameters(), method="SLSQP", jac="3-point",
        bounds=Bounds(np.zeros(16), np.ones(16)),
        constraints=LinearConstraint(constraint, np.ones(4), np.ones(4)),
        options=dict(ftol=1e-10, maxiter=500))
    if not result.success:
        raise RuntimeError(result.message)
    np.testing.assert_allclose(constraint@result.x, 1., rtol=0, atol=1e-8)
    if np.any(result.x < 0):
        raise ValueError("Circuit selection produced a negative passive weight")
    return result.x, dict(iterations=int(result.nit), evaluations=int(result.nfev),
                         objective_microstrain2=float(result.fun))


def feature_gradient_diagnosis(problem, network, parameters):
    """Locate force-loss gradients excluded by the present last-layer features."""
    gradients = []
    critical = None
    for parameter in parameters:
        eta = problem.physics.tensor(parameter)[None]
        raw = network.raw(eta).detach().requires_grad_(True)
        absolute = absolute_components(raw) if network.path_coordinates == "reference_increments" else raw
        state = problem.coordinates.decode(absolute)*problem.physics.scale
        force, _, history = problem.equilibrium(state, eta)
        residual = problem.coordinates.whiten(force)
        if np.array_equal(parameter, [-1., -1.]):
            index = int(history[0, -1].argmax())
            peak = history[0, -1, index]
            derivative = torch.autograd.grad(peak, raw, retain_graph=True)[0][0].numpy().ravel()
            critical = dict(row=len(gradients), integration_point=index, raw=raw.detach(),
                            peak_m=float(peak.detach()), derivative=derivative)
        gradients.append(torch.autograd.grad(residual.square().mean()/len(parameters), raw)[0][0].numpy())
    gradient = np.stack(gradients).reshape(len(parameters), -1)
    with torch.no_grad():
        features = (network.features(problem.physics.tensor(parameters))@network.feature_transform).numpy()
    q, _ = np.linalg.qr(features, mode="reduced")
    projected = q@(q.T@gradient)
    outside = gradient-projected
    result = dict(training_conditions=len(parameters), feature_count=features.shape[1],
        feature_singular_values=np.linalg.svd(features, compute_uv=False).tolist(),
        fraction_gradient_norm_outside_last_layer_span=float(np.linalg.norm(outside)/np.linalg.norm(gradient)),
        meaning="Projection of the current raw-state physics-loss gradient across material conditions. Hidden layers remain trainable; this locates a restriction of the present last-layer subproblem, not an irreducible model error or a stopping target.")
    if critical is not None:
        derivative, row = critical["derivative"], critical["row"]
        directions = {"independent_state": -gradient[row], "shared_last_layer": -projected[row]}
        values = {}
        for name, direction in directions.items():
            norm = np.linalg.norm(direction)
            unit = direction/norm
            analytical = float(derivative@unit)
            raw = critical["raw"]
            change = problem.physics.tensor(unit.reshape(raw.shape))
            h = 1e-5*float(raw.norm())
            peaks = []
            with torch.no_grad():
                for sign in [1, -1]:
                    trial = raw+sign*h*change
                    absolute = absolute_components(trial) if network.path_coordinates == "reference_increments" else trial
                    state = problem.coordinates.decode(absolute)*problem.physics.scale
                    history = problem.equilibrium(state, problem.physics.tensor([[-1., -1.]]))[2]
                    peaks.append(float(history[0, -1, critical["integration_point"]]))
            finite = (peaks[0]-peaks[1])/(2*h)
            np.testing.assert_allclose(analytical, finite, rtol=1e-4, atol=1e-12)
            values[name] = dict(history_derivative_per_unit_raw=analytical,
                finite_difference=finite, state_gradient_norm=float(norm),
                history_gradient_cosine=float(analytical/np.linalg.norm(derivative)))
        result["weak_closure_peak_history"] = dict(parameters=[-1., -1.], force_kn=float(problem.force[0, -1]/1000),
            integration_point=critical["integration_point"], peak_history_micro_m=critical["peak_m"]*1e6,
            damage_onset_micro_m=problem.physics.onset*1e6, directions=values,
            meaning="Local history derivative at the current critical integration point. The shared direction is the orthogonal projection of independent state-loss gradients onto the current material feature space; it is not the L-BFGS step or a proof about the full trainable network.")
    return result


@torch.no_grad()
def material_physics_diagnosis(problem, network, parameters, training_count):
    """Check whether feature expansion transfers physical equilibrium to held inputs."""
    loss = [float(problem.residual(network, problem.physics.tensor(eta)[None]).square().mean())
            for eta in parameters]
    features = (network.features(problem.physics.tensor(parameters))@network.feature_transform).numpy()
    interpolation = np.linalg.lstsq(features[:training_count].T, features[training_count:].T, rcond=None)[0].T
    np.testing.assert_allclose(interpolation@features[:training_count], features[training_count:],
                               rtol=1e-9, atol=1e-11)
    return dict(whitened_force_mean_square=loss,
                training_mean=float(np.mean(loss[:training_count])),
                held_material_mean=float(np.mean(loss[training_count:])),
                held_state_interpolation_weights=interpolation.tolist(),
                held_state_interpolation_l1=np.abs(interpolation).sum(axis=1).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="physics")
    parser.add_argument("--compare-to")
    parser.add_argument("--design-from")
    parser.add_argument("--inspect-features", action="store_true")
    parser.add_argument("--inspect-physics", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    record = json.loads((OUTPUT/f"{args.name}.json").read_text())
    setup = record["setup"]
    saved = dict(np.load(OUTPUT/f"{args.name}_query.npz"))
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    problem = MaterialHistoryProblem(setup)
    p = problem.physics
    network = FixedReferenceNetwork(problem.coordinates, problem.initial, setup["width"],
                                    setup.get("path_coordinates", "reference_increments"))
    network.load_state_dict(torch.load(OUTPUT/f"{args.name}.pt", weights_only=True)["network"])
    readout = FixedShuntReadout(uniform_parameters(), settings, p.tensor)
    rebuilt = problem.evaluate(network, saved["parameters"], readout)
    for key in saved:
        np.testing.assert_allclose(rebuilt[key], saved[key], rtol=1e-10, atol=1e-14)
    physics_training = list(range(len(setup["training_parameters"])))
    training = physics_training
    if args.design_from:
        selection_source = json.loads((OUTPUT/f"{args.design_from}_readout.json").read_text())
        selection_parameters = np.asarray(selection_source["material_parameters"])[selection_source["training_indices"]]
        training = parameter_indices(saved["parameters"], selection_parameters).tolist()
        if not set(training).issubset(physics_training):
            raise ValueError("The design selection includes held material queries")
    held = list(range(len(physics_training), len(saved["parameters"])))
    nominal = int(np.flatnonzero(np.all(saved["parameters"] == [1, -1], axis=1))[0])
    parameters, fit = select_design(saved, settings, training, nominal)
    rows = {}
    for name, design in [("uniform", uniform_parameters()), ("selected", parameters)]:
        values, circuit = fixed_circuit(saved, design, settings)
        rows[name] = dict(**design_values(design, settings),
            training=metrics(saved, design, settings, training, nominal),
            held_material_queries=metrics(saved, design, settings, held, nominal),
            twelve_bridge_power_w_range=[float(circuit["power"].sum(-1).min()), float(circuit["power"].sum(-1).max())],
            amplifier_voltage_range_v=[float(circuit["voltage"].min()), float(circuit["voltage"].max())])
    if rows["selected"]["training"]["objective_microstrain2"] > rows["uniform"]["training"]["objective_microstrain2"]:
        raise ValueError("The returned design increases the trained synthesis objective")
    selected, _ = fixed_circuit(saved, parameters, settings)
    selected_torch = FixedShuntReadout(parameters, settings, p.tensor)(p.tensor(saved["raw_gauge_strain"]))
    selected_torch = (selected_torch-selected_torch[:, :1]).numpy()
    np.testing.assert_allclose(selected, selected_torch, rtol=1e-8, atol=1e-14)
    # Full-domain parity projections diagnose the content a symmetric ansatz
    # would erase, at the same network prediction rather than another solver.
    corners = [int(np.flatnonzero(np.all(saved["parameters"] == eta, axis=1))[0])
               for eta in [[1, -1], [1, 1], [-1, -1], [-1, 1]]]
    sectors = []
    with torch.no_grad():
        states = network(p.tensor(saved["parameters"][corners])).numpy()
    increments = states[:, -1]-states[:, 0]
    for projection in problem.coordinates.projections:
        projected = (projection@(projection.T@increments.T)).T
        sectors.append((projected@problem.functionals.q_matrix.numpy().T).reshape(-1, 2, 3, 2)*1e6)
    np.testing.assert_allclose(np.sum(sectors, axis=0), saved["host_increment"][corners, -1]*1e6,
                               rtol=1e-10, atol=1e-9)
    result = dict(checkpoint=f"{args.name}.pt", query_arrays=f"{args.name}_query.npz",
        reconstruction="All saved outputs reconstructed from the CPU checkpoint",
        training_indices=training, held_material_indices=held, nominal_material_index=nominal,
        physics_training_indices=physics_training, design_sampling_source=args.design_from,
        force_kn=(saved["force_n"]/1000).tolist(), material_parameters=saved["parameters"].tolist(),
        corner_indices=corners, corner_order=["baseline", "separate_closure", "softer_top", "combined"],
        final_force_kn=float(saved["force_n"][-1]/1000),
        corner_host_final_microstrain=(saved["host_increment"][corners, -1]*1e6).tolist(),
        parity_sectors=problem.coordinates.parities, corner_parity_host_microstrain=np.asarray(sectors).tolist(),
        maximum_damage=saved["maximum_damage"].tolist(), designs=rows, fit=fit,
        gauge_strain_abs_max_microstrain=float(np.abs(saved["raw_gauge_strain"]).max()*1e6),
        interpretation=f"The complete nonlinear installed circuit is selected on {len(training)} material queries from the current PINN, using fixed-reference response values and material changes from the 30-GPa/200-GPa-per-m scenario. The {len(held)} held material queries enter neither physics training nor resistor selection. This is a conditional design on the trained fields, not a training-method gain or experimental installation validation. All physical loading steps retain the same loaded reference; the two material inputs describe fixed properties, not a re-zeroed evolving specimen.")
    if args.compare_to:
        previous = dict(np.load(OUTPUT/f"{args.compare_to}_query.npz"))
        previous_readout = json.loads((OUTPUT/f"{args.compare_to}_readout.json").read_text())
        indices = parameter_indices(saved["parameters"], previous["parameters"])
        common = {key: value if key == "force_n" else value[indices] for key, value in saved.items()}
        common_training = previous_readout["training_indices"]
        common_held = previous_readout["held_material_indices"]
        common_nominal = previous_readout["nominal_material_index"]
        np.testing.assert_array_equal(previous["force_n"], saved["force_n"])
        previous_design = np.asarray(previous_readout["designs"]["selected"]["parameters"])
        comparison = dict(source=args.compare_to, cross_evaluations={},
            common_material_parameters=previous["parameters"].tolist(),
            common_training_indices=common_training, common_held_indices=common_held)
        for field_name, fields in [("previous", previous), ("current", common)]:
            comparison["cross_evaluations"][field_name] = {
                design_name: {"training": metrics(fields, design, settings, common_training, common_nominal),
                              "held_material_queries": metrics(fields, design, settings, common_held, common_nominal)}
                for design_name, design in [("previous", previous_design), ("current", parameters)]}
        old_values, _ = fixed_circuit(saved, previous_design, settings)
        comparison["selected_readout_change_on_current_fields_rms_microstrain"] = float(
            np.sqrt(np.mean((selected[indices, 1:]-old_values[indices, 1:])**2))*1e6)
        comparison["host_change_rms_microstrain"] = float(np.sqrt(np.mean(
            (common["host_increment"][:, 1:]-previous["host_increment"][:, 1:])**2))*1e6)
        result["training_comparison"] = comparison
    if args.inspect_features:
        result["feature_gradient"] = feature_gradient_diagnosis(problem, network, setup["training_parameters"])
    if args.inspect_physics:
        result["material_physics"] = material_physics_diagnosis(problem, network, saved["parameters"], len(physics_training))
        interpolation = np.asarray(result["material_physics"]["held_state_interpolation_weights"])
        interpolated_host = np.einsum("ij,j...->i...", interpolation, saved["host_increment"][physics_training])
        np.testing.assert_allclose(interpolated_host, saved["host_increment"][held], rtol=1e-9, atol=1e-13)
        if args.compare_to:
            previous_setup = json.loads((OUTPUT/f"{args.compare_to}.json").read_text())["setup"]
            if any(previous_setup[key] != setup[key] for key in ("domain", "material_configuration")):
                raise ValueError("Physics comparison requires the same material domain and configuration")
            previous_network = FixedReferenceNetwork(problem.coordinates, problem.initial, previous_setup["width"],
                                                     previous_setup.get("path_coordinates", "reference_increments"))
            previous_network.load_state_dict(torch.load(OUTPUT/f"{args.compare_to}.pt", weights_only=True)["network"])
            result["material_physics"]["previous"] = dict(checkpoint=f"{args.compare_to}.pt",
                material_parameters=saved["parameters"].tolist(),
                sampling="Both checkpoints evaluated on the current training grid and held set",
                **material_physics_diagnosis(problem, previous_network, saved["parameters"], len(physics_training)))
    (OUTPUT/f"{args.name}_readout.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(OUTPUT/f"{args.name}_readout.npz", parameters=parameters,
        selected_increment=selected, uniform_increment=saved["circuit_increment"],
        host_increment=saved["host_increment"], material_parameters=saved["parameters"])
    print(json.dumps(dict(corners=result["corner_host_final_microstrain"],
        designs={key: {part: row[part] for part in ("training", "held_material_queries")}
                 for key, row in rows.items()}, fit=fit)), flush=True)


if __name__ == "__main__":
    main()
