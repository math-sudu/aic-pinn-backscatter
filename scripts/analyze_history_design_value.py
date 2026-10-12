"""Relate saved low-load PINN/circuit changes to regional monitoring outputs.

Reuse the common local physics correction and installed circuits from B2 5.38.
Material comparisons, temporal increments and spatial contrasts remain separate.
No mechanical solve, training or resistor selection is performed here.
"""

import json

import numpy as np

from analyze_fixed_reference_history import fixed_circuit, metrics
from fixed_reference_history import ROOT, OUTPUT, parameter_indices
from weighted_readout import shunt_design, uniform_parameters


NAMES = ("physics_history_metric", "physics_regional_control", "regional_history")


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def layers(components):
    """Invert mean and half top-minus-bottom difference, retaining signs."""
    return np.stack([components[..., 0]-components[..., 1],
                     components[..., 0]+components[..., 1]], axis=-1)


def monitoring_features(values):
    """Shape: material, step, direction, region, (mean, half difference)."""
    return dict(layer_strain=layers(values)[:, 1:],
                depth_half_difference=values[:, 1:, ..., 1],
                right_minus_left=values[:, 1:, :, 2]-values[:, 1:, :, 0],
                loading_increment=np.diff(values, axis=1))


def feature_comparison(predicted, target):
    result = {}
    predicted_features = monitoring_features(predicted)
    for name, truth in monitoring_features(target).items():
        estimate = predicted_features[name]
        mismatches = np.argwhere(np.sign(estimate) != np.sign(truth))
        result[name] = dict(
            sign_disagreements=[dict(index=index.tolist(),
                host_microstrain=float(truth[tuple(index)]),
                readout_microstrain=float(estimate[tuple(index)])) for index in mismatches],
            rms_microstrain=rms(truth),
            minimum_absolute_host_microstrain=float(np.min(np.abs(truth))))
    host_rank = np.argmax(target[:, 1:, ..., 0], axis=-1)
    output_rank = np.argmax(predicted[:, 1:, ..., 0], axis=-1)
    result["largest_regional_mean"] = dict(
        host_region_indices=np.unique(host_rank).tolist(),
        readout_region_indices=np.unique(output_rank).tolist(),
        disagreements=np.argwhere(host_rank != output_rank).tolist())
    return result


def history_noise_covariance(component_variance, steps):
    """Independent equal-variance conversions, with one shared reference read.

    Each nonreference sample is e_n-e_0. Temporal contrasts must use the
    resulting off-diagonal covariance; the reference cancels in differences.
    """
    return (np.eye(steps)+np.ones((steps, steps)))[..., None, None, None]*component_variance


def direct_bridge_noise_context(parameters, settings, steps):
    """Ideal unloaded bridge slope and a declared ADC-only noise scenario.

    This is not an ADC input-load or installed-system model. Each region has
    its actual shunts and leads. One independent conversion at each depth and
    state, including the loaded reference, uses the archived short-input RMS.
    """
    _, rho, completion = shunt_design(parameters, settings)
    slope = (settings["excitation_v"]*settings["gauge_factor"]*
             settings["sensor_resistance_ohm"]*(rho*rho).sum(-1)/(4*completion))
    adc = settings["digital_alternative"]
    scenarios = []
    for rate, sigma in zip(adc["data_rates_sps"], adc["shorted_input_noise_rms_v"]):
        layer_variance = ((sigma/slope)*1e6).reshape(2, 3, 2)**2
        # Mean and half-difference each use coefficients (+/- 1/2, 1/2).
        variance = np.repeat(layer_variance.sum(-1, keepdims=True)/4, 2, axis=-1)
        covariance = history_noise_covariance(variance, steps)
        scenarios.append(dict(data_rate_sps=rate,
            shorted_input_noise_rms_microvolt=sigma*1e6,
            single_state_component_variance_microstrain2=variance.tolist(),
            fixed_reference_increment_rms_microstrain=np.sqrt(2*variance).tolist(),
            consecutive_increment_rms_microstrain=np.sqrt(
                covariance[0, 0]+covariance[1, 1]-2*covariance[0, 1]).tolist()))
    return dict(unloaded_slope_microvolt_per_microstrain=slope.reshape(2, 3, 2).tolist(),
                scenarios=scenarios)


def representation_effect(readout, saved):
    """Counterfactual removal of existing reflection sectors, not retraining."""
    sectors = np.asarray(readout["corner_parity_host_microstrain"])
    full = sectors.sum(axis=0)
    expected = saved["host_increment"][readout["corner_indices"], -1]*1e6
    np.testing.assert_allclose(full, expected, rtol=1e-10, atol=1e-9)
    even_index = readout["parity_sectors"].index([1, 1])
    even = sectors[even_index]
    return dict(parameters=saved["parameters"][readout["corner_indices"]].tolist(),
        right_minus_left_microstrain=(full[:, :, 2]-full[:, :, 0]).tolist(),
        even_only_right_minus_left_microstrain=(even[:, :, 2]-even[:, :, 0]).tolist(),
        omitted_component_rms_microstrain=rms(full-even),
        omitted_component_max_abs_microstrain=float(np.max(np.abs(full-even))))


def design_cross_evaluation(fields, designs, settings, groups, nominal):
    """Evaluate existing candidates; held queries never select new resistors."""
    result = {}
    for field_name, data in fields.items():
        result[field_name] = {}
        for group, rows in groups.items():
            values = {name: metrics(data, design, settings, rows, nominal)
                      for name, design in designs.items()}
            contrast = (values["regional_history"]["objective_microstrain2"]-
                        values["physics_history_metric"]["objective_microstrain2"])
            result[field_name][group] = dict(designs=values,
                ranking=sorted(values, key=lambda name: values[name]["objective_microstrain2"]),
                regional_minus_starting_objective_microstrain2=contrast)
    return result


def analyze():
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    reference = dict(np.load(OUTPUT/"regional_history_reference.npz"))
    source = json.loads((OUTPUT/"regional_history_reference.json").read_text())
    comparison = json.loads((OUTPUT/"regional_history_comparison.json").read_text())
    host = reference["host_increment"]*1e6
    groups = {name: source[f"{name}_indices"] for name in ("corner", "held")}
    nominal = source["nominal_index"]
    circuits, networks, installed = {}, {}, None
    fields, designs = {"common_fields": reference}, {}
    unloaded_settings = dict(settings, input_differential_resistance_ohm=np.inf,
        input_common_mode_resistance_ohm=np.inf, input_bias_a=[0., 0.],
        input_switch_resistance_ohm=0., amplifier_gain=1., amplifier_reference_v=0.)
    for name in (*NAMES, "uniform"):
        if name == "uniform":
            parameters = uniform_parameters()
        else:
            readout = json.loads((OUTPUT/f"{name}_readout.json").read_text())
            parameters = np.asarray(readout["designs"]["selected"]["parameters"])
            designs[name] = parameters
        values, circuit = fixed_circuit(reference, parameters, settings)
        values = values*1e6
        if installed is None:
            installed = values
        synthesis = {group: metrics(reference, parameters, settings, rows, nominal)
                     for group, rows in groups.items()}
        if name != "uniform":
            for group, recorded_group in (("corner", "corners"), ("held", "held")):
                recorded = comparison["models"][name]["selected_design_on_common_fields"][recorded_group]
                np.testing.assert_allclose(list(synthesis[group].values()), list(recorded.values()),
                                           rtol=1e-10, atol=1e-12)
        noise = direct_bridge_noise_context(parameters, settings, host.shape[1]-1)
        direct_values, direct_circuit = fixed_circuit(reference, parameters, unloaded_settings)
        direct_values *= 1e6
        np.testing.assert_allclose(direct_circuit["gain"].reshape(2, 3, 2),
            noise["unloaded_slope_microvolt_per_microstrain"], rtol=1e-12, atol=1e-14)
        bias = (direct_values-host)[:, 1:]
        # Online observation is a fixed-reference response. The offline
        # material-scenario penalty is not another physical noisy time sample.
        expected_operating = []
        for scenario in noise["scenarios"]:
            noise_variance = np.asarray(scenario["fixed_reference_increment_rms_microstrain"])**2
            expected_operating.append(dict(data_rate_sps=scenario["data_rate_sps"],
                **{group: dict(synthesis_rms_microstrain=rms(bias[rows]),
                    noise_rms_microstrain=float(np.sqrt(noise_variance.mean())),
                    combined_rms_microstrain=float(np.sqrt(np.mean(bias[rows]**2)+noise_variance.mean())))
                    for group, rows in groups.items()}))
        circuits[name] = dict(
            synthesis=synthesis,
            features_on_common_fields=feature_comparison(values, host),
            change_from_starting_design=dict(
                rms_microstrain=rms((values-installed)[:, 1:]),
                max_abs_microstrain=float(np.max(np.abs((values-installed)[:, 1:]))),
                final_by_direction_region_component_rms_microstrain=
                    np.sqrt(np.mean((values[:, -1]-installed[:, -1])**2, axis=0)).tolist()),
            loaded_input_slope_microvolt_per_microstrain=(circuit["gain"]/settings["amplifier_gain"]).reshape(2, 3, 2).tolist(),
            direct_bridge_noise_context=noise,
            direct_vs_loaded_response_max_abs_microstrain=float(np.max(np.abs(direct_values-values))),
            conditional_operating_response=expected_operating)
        if name != "uniform":
            saved = dict(np.load(OUTPUT/f"{name}_query.npz"))
            indices = parameter_indices(saved["parameters"], reference["parameters"])
            fields[name] = {key: value if key == "force_n" else value[indices]
                            for key, value in saved.items()}
            network_host = saved["host_increment"][indices]*1e6
            networks[name] = dict(
                host_features=feature_comparison(network_host, host),
                reflection_sector_removal=representation_effect(readout, saved))
    material_change = host[np.arange(len(host)) != nominal]-host[nominal]
    return dict(
        source="B2 5.38 saved common fields and selected full nonlinear circuits",
        reference=source["definition"],
        field_directory="results/ideation/stateful_pilot/fixed_reference_history",
        inputs=["regional_history_reference.npz", "regional_history_comparison.json",
                *[f"{name}_{suffix}" for name in NAMES for suffix in ("query.npz", "readout.json")]],
        force_kn=(reference["force_n"]/1000).tolist(),
        material_parameters=reference["parameters"].tolist(),
        groups=groups,
        nominal_material_index=nominal,
        axes=dict(direction=["d1", "d2"], region=["left", "central", "right"],
                  component=["mean", "half_top_minus_bottom"], layer=["bottom", "top"]),
        feature_index_definition="material, nonreference step or loading interval, direction, then region/layer or component as applicable; indices are zero based",
        signal_scale=dict(response_rms_microstrain=rms(host[:, 1:]),
            material_change_rms_microstrain=rms(material_change[:, 1:]),
            loading_increment_rms_microstrain=rms(np.diff(host, axis=1)),
            final_min_by_direction_region_component_microstrain=host[:, -1].min(axis=0).tolist(),
            final_max_by_direction_region_component_microstrain=host[:, -1].max(axis=0).tolist(),
            final_right_minus_left_mean_range_microstrain=[
                float(np.min(host[:, -1, :, 2, 0]-host[:, -1, :, 0, 0])),
                float(np.max(host[:, -1, :, 2, 0]-host[:, -1, :, 0, 0]))]),
        circuits=circuits, networks=networks,
        design_cross_evaluation=design_cross_evaluation(fields, designs, settings, groups, nominal),
        noise_context=dict(source="references/hardware/ti_ads1220/full.md Table 7-1; inputs/regional_readout.json digital_alternative",
            assumptions="Unloaded bridge with the same shunts and leads; direct ADC; independent equal-variance conversions at each depth and time, including one shared reference reading; fixed zero-state gain. The archived normal-mode 3.3-V/PGA-128/internal-2.048-V-reference short-input noise provides scale, not installed-system noise. No temporal averaging, input-current loading, sensor noise or RF loss is modeled.",
            history_covariance="Cov(y_n, y_m) = v*(delta_nm + 1), where v is the single-state regional component variance. Adjacent differences cancel reference noise; material scenarios are distinct conditions, not a time sequence.",
            use="Conditional operating-response RMS combines synthesis bias and this noise scenario, not the offline value-plus-material-change objective. No detection threshold or design rejection is inferred."))


def main():
    result = analyze()
    path = ROOT/"results/ideation/B2_history_design_value.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(path),
        held_design_order={name: row["held"]["ranking"]
                           for name, row in result["design_cross_evaluation"].items()},
        new_circuit_change=result["circuits"]["regional_history"]["change_from_starting_design"]["rms_microstrain"])), flush=True)


if __name__ == "__main__":
    main()
