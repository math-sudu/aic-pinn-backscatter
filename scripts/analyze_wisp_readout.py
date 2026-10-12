"""Relate the saved acquisition load to the pinned WISP hardware and firmware."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def analyze(readout, regional, wisp):
    """Charge accounting for linear rails; radio framing is a subtotal only."""
    vhi = regional["storage_voltage_high_v"]
    vlo = regional["storage_voltage_low_v"]
    drop = vhi - vlo
    vmean = (vhi + vlo) / 2
    adc_supply = regional["digital_alternative"]["supply_v"]
    stock_voltage = wisp["stock_regulated_voltage_v"]
    cap_energy = (0.5 * wisp["stock_charge_pump_capacitance_f"]
                  * wisp["charge_pump_nominal_release_voltage_v"]**2)
    fragments = readout["transport"]["epc_fragments_per_snapshot"]
    frame_bits = 8*wisp["epc_bytes"] + wisp["pc_bits"] + wisp["crc_bits"]
    symbols = frame_bits + wisp["fm0_preamble_symbols"] + wisp["fm0_end_symbols"]
    tx_times = [fragments*(symbols+extra)/wisp["backscatter_link_frequency_hz"]
                for extra in (0, wisp["fm0_extended_pilot_symbols"])]
    tx_charge = [t*wisp["controller_transmit_current_a_scenario"] for t in tx_times]

    designs = {}
    for name, design in readout["designs"].items():
        scans = []
        for scan in design["acquisition"]["scans"]:
            times = [t/1000 for t in scan["scan_ms_range"]]
            base_cap = scan["linear_pass_storage_mf_range"]
            controller_charge = [wisp["controller_scan_current_a_scenario"]*t
                                 for t in times]
            acquisition_cap = [c+1000*q/drop for c, q in zip(base_cap, controller_charge)]
            scans.append({
                "data_rate_sps": scan["data_rate_sps"],
                "scan_ms_range": scan["scan_ms_range"],
                "readout_only_linear_storage_mf_range": base_cap,
                "readout_plus_controller_linear_storage_mf_range": acquisition_cap,
                "readout_plus_controller_reservoir_energy_mj_range": [
                    c*drop*vmean for c in acquisition_cap],
                "storage_increment_mf_per_100ua_during_scan": [
                    1000*0.0001*t/drop for t in times],
                "rail_energy_over_47uf_at_2p4v_range": [
                    e/1000/cap_energy for e in scan["acquisition_rail_energy_mj_range"]],
                "recharge_seconds_at_1mw_net_for_acquisition_scenario": [
                    c*drop*vmean for c in acquisition_cap],
            })
        designs[name] = scans

    return {
        "role": "Board compatibility, acquisition-current scaling and EPC airtime; no new PINN, circuit selection or physical solve",
        "settings": "inputs/wisp_readout.json",
        "source_acquisition": "results/ideation/B2_digital_readout.json",
        "stock_board": {
            "regulated_voltage_v": stock_voltage,
            "adc_minimum_digital_supply_v": wisp["adc_digital_supply_min_v"],
            "adc_required_input_high_v": adc_supply*wisp["adc_input_high_supply_fraction"],
            "mcu_nominal_rail_minus_adc_required_high_v": (
                stock_voltage-adc_supply*wisp["adc_input_high_supply_fraction"]),
            "adc_rail_minus_mcu_input_absolute_max_v": (
                adc_supply-stock_voltage-wisp["mcu_input_absolute_headroom_v"]),
            "charge_pump_cap_energy_at_named_voltage_mj": cap_energy*1000,
            "stock_user_bank_bytes": wisp["stock_user_bank_bytes"],
            "raw_snapshot_bytes": readout["transport"]["raw_adc_payload_bytes"],
        },
        "designs": designs,
        "epc_transmission_subtotal": {
            "fragments": fragments,
            "frame_data_bits_per_fragment": frame_bits,
            "short_preamble_symbols_per_fragment": symbols,
            "extended_preamble_symbols_per_fragment": symbols+wisp["fm0_extended_pilot_symbols"],
            "snapshot_tx_ms_short_extended": [1000*t for t in tx_times],
            "controller_tx_charge_uc_scenario_short_extended": [q*1e6 for q in tx_charge],
            "linear_storage_increment_mf_scenario_short_extended": [1000*q/drop for q in tx_charge],
        },
        "receive_and_persistence_scaling": {
            "storage_increment_mf_per_100ms_at_receive_current_scenario": (
                1000*wisp["controller_receive_current_a_scenario"]*0.1/drop),
            "reservoir_energy_mj_per_100ms_at_receive_current_scenario": (
                1000*wisp["controller_receive_current_a_scenario"]*0.1*vmean),
            "storage_increment_mf_per_microcoulomb_extra_charge": 0.001/drop,
            "recharge_relation": "t >= E_cycle/P_net for constant positive net power delivered to the reservoir; E_cycle includes reception, persistence, supply and leakage costs",
        },
        "scope": {
            "hardware": wisp["board_scope"],
            "controller_currents": wisp["controller_current_basis"],
            "rf_time": wisp["rf_time_basis"],
            "stock_energy": wisp["stock_energy_basis"],
            "power": wisp["energy_architecture"],
            "transport": "ACK callback follows local EPC transmission. A completed snapshot remains replayable until an application confirmation names its node, context and sequence.",
        },
    }


def main():
    def read(path):
        return json.loads((ROOT/path).read_text(encoding="utf-8"))

    result = analyze(read("results/ideation/B2_digital_readout.json"),
                     read("inputs/regional_readout.json"),
                     read("inputs/wisp_readout.json"))
    output = ROOT/"results/ideation/B2_wisp_readout.json"
    output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"stock_board": result["stock_board"],
                      "regional": result["designs"]["regional"],
                      "tx": result["epc_transmission_subtotal"]}, indent=2))


if __name__ == "__main__":
    main()
