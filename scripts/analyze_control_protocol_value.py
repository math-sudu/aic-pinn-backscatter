"""Relate the saved control-history designs to sampled regional responses.

Physical fields and selected resistors are reused. The direct ADC scenario
retains the installed strain kernel, shunts, leads and fixed calibration.
It evaluates conversion noise and twelve-channel conversion energy, without
performing a mechanical solve, neural update or new design selection.
"""

import json

import numpy as np
import torch

from analyze_control_protocol_design import STATES, circuit_values, metrics
from analyze_history_design_value import direct_bridge_noise_context, rms
from controlled_fracture_history import ControlledFractureProblem, ROOT
from weighted_readout import weighted_response


def direct_settings(settings):
    return dict(settings, input_differential_resistance_ohm=np.inf,
                input_common_mode_resistance_ohm=np.inf, input_bias_a=[0., 0.],
                input_switch_resistance_ohm=0., amplifier_gain=1.,
                amplifier_reference_v=0.)


def conversion_scan(circuit, settings):
    """One serial ADC, twelve layers; conversion-window energy only.

    A held structural state is assumed throughout a scan. Channel excitation
    gating is an ideal schedule whose physical switches are not modeled here.
    Both schedules include the same ADC active power and actual bridge power.
    """
    adc = settings["digital_alternative"]
    channel_count = circuit["power"].shape[-1]
    total_power = circuit["power"].sum(-1)
    adc_power = adc["supply_v"]*(adc["analog_current_a"]+adc["digital_current_a"])
    storage_swing = settings["storage_voltage_high_v"]**2-settings["storage_voltage_low_v"]**2
    scans = []
    for rate, cycles in zip(adc["data_rates_sps"], adc["single_shot_conversion_cycles"]):
        conversion = np.array([cycles/adc["conversion_clock_hz"],
            cycles/adc["conversion_clock_hz"]+adc["internal_oscillator_wakeup_max_s"]])
        scan = channel_count*conversion
        schedules = {}
        for name, bridge_time in (("all_bridges_on", scan), ("selected_bridge_only", conversion)):
            energy = total_power[..., None]*bridge_time+adc_power*scan
            schedules[name] = dict(conversion_energy_mj_range=[float(energy.min()*1e3), float(energy.max()*1e3)],
                ideal_storage_capacitance_mf_range=[float(2*energy.min()/storage_swing*1e3),
                                                   float(2*energy.max()/storage_swing*1e3)])
        scans.append(dict(data_rate_sps=rate, channel_conversion_ms_range=(conversion*1e3).tolist(),
            twelve_channel_scan_ms_range=(scan*1e3).tolist(), schedules=schedules))
    return dict(channels=channel_count, adc_active_power_mw=adc_power*1e3,
        simultaneous_bridge_power_mw_range=[float(total_power.min()*1e3), float(total_power.max()*1e3)],
        single_bridge_power_mw_range=[float(circuit["power"].min()*1e3), float(circuit["power"].max()*1e3)],
        scans=scans)


def operating_response(data, parameters, settings):
    direct = direct_settings(settings)
    values, circuit = circuit_values(data, parameters, direct)
    loaded, _ = circuit_values(data, parameters, settings)
    bias = values-data["host"]
    history_bias = bias[:, 3]-bias[:, 1]
    noise = direct_bridge_noise_context(parameters, settings, 3)
    np.testing.assert_allclose(circuit["gain"].reshape(2, 3, 2),
        noise["unloaded_slope_microvolt_per_microstrain"], rtol=1e-12, atol=1e-14)
    scenarios = []
    for scenario in noise["scenarios"]:
        variance = np.asarray(scenario["fixed_reference_increment_rms_microstrain"])**2
        # The original reference cancels in return-minus-first. Each of the
        # two passages contributes one conversion per layer, hence the same 2v.
        scenarios.append(dict(data_rate_sps=scenario["data_rate_sps"],
            response_or_history_noise_rms_microstrain=float(np.sqrt(variance.mean())),
            response_combined_rms_microstrain=float(np.sqrt(np.mean(bias[:, 1:]**2)+variance.mean())),
            history_combined_rms_microstrain=float(np.sqrt(np.mean(history_bias**2)+variance.mean()))))
    nodes = circuit["nodes"]
    common = nodes.mean(-1)
    differential = np.abs(nodes[..., 0]-nodes[..., 1])
    # ADS1220 Eq. 12--14 also requires Vcm >= AVDD/4 for AVSS = 0.
    pga_swing = settings["digital_alternative"]["pga_gain"]*differential/2
    adc = settings["digital_alternative"]
    summary = dict(response_synthesis_rms_microstrain=rms(bias[:, 1:]),
        history_synthesis_rms_microstrain=rms(history_bias),
        direct_vs_original_loaded_max_abs_microstrain=float(np.max(np.abs(values-loaded))),
        direct_adc_noise=noise, scenarios=scenarios,
        input_differential_abs_max_mv=float(differential.max()*1e3),
        input_common_mode_v_range=[float(common.min()), float(common.max())],
        pga_headroom_min_v=float(np.minimum(np.minimum(common-.2-pga_swing,
            adc["supply_v"]-.2-pga_swing-common), common-adc["supply_v"]/4).min()),
        differential_full_scale_mv=adc["internal_reference_v"]/adc["pga_gain"]*1e3,
        host_history_microstrain=(data["host"][:, 3]-data["host"][:, 1]).tolist(),
        circuit_history_microstrain=(values[:, 3]-values[:, 1]).tolist())
    return summary, values


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    directory = ROOT/"results/ideation"
    source = json.loads((directory/"B2_unloading_history.json").read_text())
    problem = ControlledFractureProblem(source["configuration"])
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    design_record = json.loads((directory/"B2_control_protocol_design.json").read_text())
    sources = ["B2_unloading_history", "B2_control_history_lower_peak", "B2_control_history_mid_peak"]
    physical, all_raw = [], []
    for name in sources:
        saved = np.load(directory/(name+".npz"))
        fields = problem.physics.tensor(saved["displacement_m"][:33])
        raw = (fields@problem.base.raw_matrix.T).numpy()
        host = ((fields-fields[:1])@problem.base.functionals.q_matrix.T).numpy().reshape(-1, 2, 3, 2)*1e6
        physical.append(dict(raw=raw[STATES], host=host[STATES]))
        all_raw.append(raw)
    groups = dict(training={key: np.stack([row[key] for row in physical[:2]]) for key in ("raw", "host")},
                  held={key: physical[2][key][None] for key in ("raw", "host")})
    designs, values = {}, {}
    for name, design in design_record["designs"].items():
        parameters = np.asarray(design["parameters"])
        by_group, values[name] = {}, {}
        for group, data in groups.items():
            # Reconstruction targets the original design consequence, rather
            # than imposing an accuracy target on a new physical prediction.
            actual = metrics(data, parameters, settings)
            recorded = design_record["cross_evaluation"]["physical_evaluation"][group][name]
            np.testing.assert_allclose(list(actual.values()), list(recorded.values()), rtol=1e-8, atol=1e-8)
            by_group[group], values[name][group] = operating_response(data, parameters, settings)
        circuit = weighted_response(np.stack(all_raw), parameters, direct_settings(settings))
        designs[name] = dict(operating_response=by_group,
                            acquisition=conversion_scan(circuit, settings))
    changes = {}
    for group in groups:
        difference = values["regional"][group]-values["physics"][group]
        changes[group] = dict(output_rms_microstrain=rms(difference[:, 1:]),
            history_rms_microstrain=rms(difference[:, 3]-difference[:, 1]),
            history_change_by_component_microstrain=(difference[:, 3]-difference[:, 1]).tolist())
        recorded = design_record["physical_output_change"][group]
        np.testing.assert_allclose([changes[group]["output_rms_microstrain"], changes[group]["history_rms_microstrain"]],
            [recorded["regional_minus_physics_output_rms_microstrain"], recorded["regional_minus_physics_history_rms_microstrain"]],
            rtol=1e-7, atol=1e-7)
    report = dict(
        role="Engineering interpretation of saved control-history PINN and fixed-circuit design increments",
        physical_sources=["results/ideation/"+name+".npz" for name in sources],
        design_source="results/ideation/B2_control_protocol_design.json",
        sensor_assumption="Existing 64 gauges, uniform 30-mm strain transfer, 120-ohm/GF-1.8 sensors, fixed shunts and leads; no installed transfer calibration is available",
        adc_assumption="Direct unloaded bridge, one independent conversion per layer and time; one common loaded reference; archived ADS1220 Table 7-1 short-input noise, normal mode/PGA 128/3.3 V/internal 2.048 V reference; no input-current loading or environmental noise",
        scan_assumption="One serial ADC and twelve differential channels at a held mechanical state. Table 8-4 single-shot time plus 0--50 us oscillator wake-up at nominal clock. Values cover the reference and all 33 physical states of each protocol. Energy includes bridge and active ADC only; additional settling, switches, controller, regulators and backscatter are excluded. Storage capacitance uses ideal conversion from 5.0 to 4.5 V.",
        original_design_evaluation_reconstructed=True,
        full_history_gauge_abs_max_microstrain=float(np.max(np.abs(all_raw))*1e6),
        regional_minus_physics_direct_output=changes,
        designs=designs)
    output = directory/"B2_control_protocol_value.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({name: value["operating_response"]["held"]["scenarios"]
                      for name, value in designs.items()}), flush=True)


if __name__ == "__main__":
    main()
