"""Re-evaluate saved PINN circuit choices through the switched digital interface."""

import json

import numpy as np
import torch

from analyze_control_protocol_design import STATES
from analyze_control_protocol_value import direct_settings
from analyze_history_design_value import rms
from controlled_fracture_history import ControlledFractureProblem, ROOT
from digital_readout import adc_codes, decode_snapshot, encode_snapshot, ratio_response, scan_budget
from weighted_readout import components, weighted_response


def evaluation(circuit, host, adc):
    readout = (circuit["readout"]-circuit["readout"][:, :1])*1e6
    error = readout-host
    history = error[:, 3]-error[:, 1]
    objective = np.mean(error[:, 1:]**2)+np.mean(history**2)
    if len(readout) == 2:
        objective += np.mean((history[0]-history[1])**2)
    scale = 2**23*adc["pga_gain"]
    codes = adc_codes(circuit["ratio"], adc["pga_gain"])
    quantized = components((codes/scale-circuit["zero"])/circuit["gain"])
    quantized = (quantized-quantized[:, :1])*1e6
    scenarios = []
    for rate, sigma in zip(adc["data_rates_sps"], adc["shorted_input_noise_rms_v"]):
        # A state-dependent external reference changes volts per strain.
        layer_variance = (sigma/(circuit["reference"]*circuit["gain"])*1e6)**2
        variance = np.repeat(layer_variance.reshape(*layer_variance.shape[:-1], 2, 3, 2).sum(-1, keepdims=True)/4, 2, axis=-1)
        increment_variance = variance[:, 1:]+variance[:, :1]
        history_variance = variance[:, 3]+variance[:, 1]
        scenarios.append(dict(data_rate_sps=rate,
            conditional_response_rms_microstrain=float(np.sqrt(np.mean(error[:, 1:]**2)+increment_variance.mean())),
            conditional_history_rms_microstrain=float(np.sqrt(np.mean(history**2)+history_variance.mean()))))
    return dict(response_synthesis_rms_microstrain=rms(error[:, 1:]),
        history_synthesis_rms_microstrain=rms(history),
        objective_root_microstrain=float(np.sqrt(objective)),
        quantization_response_change_rms_microstrain=rms((quantized-readout)[:, 1:]),
        conditional_input_noise=scenarios), readout, codes


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    directory = ROOT/"results/ideation"
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    interface = json.loads((ROOT/"inputs/digital_readout.json").read_text())
    sources = ["B2_unloading_history", "B2_control_history_lower_peak", "B2_control_history_mid_peak"]
    source = json.loads((directory/(sources[0]+".json")).read_text())
    problem = ControlledFractureProblem(source["configuration"])
    raw, host = [], []
    for name in sources:
        fields = problem.physics.tensor(np.load(directory/(name+".npz"))["displacement_m"][:33])
        raw.append((fields@problem.base.raw_matrix.T).numpy())
        host.append(((fields-fields[:1])@problem.base.functionals.q_matrix.T).numpy().reshape(-1, 2, 3, 2)*1e6)
    raw, host = np.stack(raw), np.stack(host)[:, STATES]
    designs = json.loads((directory/"B2_control_protocol_design.json").read_text())["designs"]
    prior = json.loads((directory/"B2_control_protocol_value.json").read_text())
    report, values = {}, {}
    for name, design in designs.items():
        parameters = np.asarray(design["parameters"])
        circuit = ratio_response(raw, parameters, settings, interface)
        selected = {key: value[:, STATES] if key in (
            "ratio", "readout", "reference") else value for key, value in circuit.items()}
        groups = {}
        values[name] = {}
        for group, indices in (("training", [0, 1]), ("held", [2])):
            group_circuit = {key: value[indices] if key in (
                "ratio", "readout", "reference") else value for key, value in selected.items()}
            groups[group], values[name][group], codes = evaluation(group_circuit, host[indices], settings["digital_alternative"])
            direct = weighted_response(raw[indices][:, STATES], parameters, direct_settings(settings))["readout"]
            direct = (direct-direct[:, :1])*1e6
            np.testing.assert_allclose(rms((direct-host[indices])[:, 1:]),
                prior["designs"][name]["operating_response"][group]["response_synthesis_rms_microstrain"], rtol=1e-8)
            groups[group]["interface_minus_direct_response_rms_microstrain"] = rms((values[name][group]-direct)[:, 1:])
        packets = encode_snapshot(codes[0, -1], node_id=68, reference_id=1, sequence=3)
        identity, rebuilt = decode_snapshot(packets[::-1]+packets[:1])
        np.testing.assert_array_equal(rebuilt, codes[0, -1])
        current_shift = dict(interface, adc_input_current_scenario_a=[
            interface["adc_input_current_scenario_a"][0]+1e-9,
            interface["adc_input_current_scenario_a"][1]])
        shifted = ratio_response(raw[2, 32], parameters, settings, current_shift,
                                 calibration=(circuit["zero"], circuit["gain"]))
        drift = components((shifted["ratio"]-circuit["ratio"][2, 32])/circuit["gain"])*1e6
        nodes = circuit["nodes"]
        differential = np.abs(nodes[..., 0]-nodes[..., 1])
        common = nodes.mean(-1)
        adc = settings["digital_alternative"]
        swing = adc["pga_gain"]*differential/2
        headroom = np.minimum(np.minimum(common-.2-swing, adc["supply_v"]-.2-swing-common), common-adc["supply_v"]/4)
        report[name] = dict(evaluation=groups,
            calibration=dict(zero_ratio=circuit["zero"].tolist(), ratio_per_strain=circuit["gain"].tolist()),
            reference_v_range=[float(circuit["reference"].min()), float(circuit["reference"].max())],
            input_common_mode_v_range=[float(common.min()), float(common.max())],
            input_differential_abs_max_mv=float(differential.max()*1e3),
            pga_headroom_min_v=float(headroom.min()),
            positive_input_current_drift_one_na_microstrain=drift.tolist(),
            acquisition=scan_budget(circuit, settings, interface),
            packet_example=dict(identity=identity, twelve_epc_hex=[packet.hex() for packet in packets]))
    changes = {}
    for group in ("training", "held"):
        difference = values["regional"][group]-values["physics"][group]
        changes[group] = dict(response_rms_microstrain=rms(difference[:, 1:]),
                             history_rms_microstrain=rms(difference[:, 3]-difference[:, 1]))
    output = dict(role="Switched ratiometric acquisition of existing fixed circuit designs; no new training, physical solve or resistor selection",
        sources=["results/ideation/"+name+".npz" for name in sources],
        settings="inputs/digital_readout.json", designs=report,
        regional_minus_physics=changes,
        held_ranking=sorted(report, key=lambda name: report[name]["evaluation"]["held"]["objective_root_microstrain"]),
        transport=dict(epc_bytes=12, epc_fragments_per_snapshot=12, raw_adc_payload_bytes=36,
            transmitted_epc_payload_bytes=144,
            implementation="Host-side codec and assembly checks. WISP5 EPC sensor example supplies the firmware path; this is not a flashed WISP/ADS1220 driver or measured RF exchange.",
            reference="Node/reference/sequence key prevents mixing channel fragments. Coefficients and loaded reference codes stay at the receiver. Sequence reservation and incomplete acquisition invalidation must persist across brownouts; identity is never reused for a different snapshot."),
        remaining_dependencies=["Installed strain transfer and per-channel calibration",
            "Voltage/temperature-dependent ADC input currents and complete external-reference noise",
            "Physical cable and analog/excitation thermal settling",
            "Controller GPIO/SPI voltage interface and persistent WISP firmware integration",
            "Controller/RF/retry/regulator/storage overhead and harvested power at the installation"],
        source_scope="All three protocols and 33 saved states per protocol are reused for voltage and acquisition bounds. The 0.75-mm protocol remains excluded from training and selection. Timing assumes a held structural state; packets transmit an already completed snapshot.")
    path = directory/"B2_digital_readout.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(held={name: value["evaluation"]["held"] for name, value in report.items()},
        acquisition=report["regional"]["acquisition"], ranking=output["held_ranking"])), flush=True)


if __name__ == "__main__":
    main()
