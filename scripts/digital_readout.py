"""Twelve switched bridges with a selected excitation reference and fixed gain."""

import struct

import numpy as np

from weighted_readout import components, shunt_design


def switched_bridge(rs, rc, settings, interface):
    """DC solution including excitation, signal and reference switch drops.

    Positive input currents leave the corresponding node. Each shared bus
    includes the selected mux drain and the other two disabled bank drains.
    The declared currents are scenarios, not ADS1220 worst-case ratings.
    """
    rs, rc = np.asarray(rs), np.asarray(rc)
    ron = interface["mux_on_resistance_ohm"]
    path = ron + interface["signal_series_filter_ohm"]
    excitation_path = ron+interface["excitation_series_ohm"]
    reference_path = ron+interface["reference_series_ohm"]
    ip, im = (np.asarray(interface["adc_input_current_scenario_a"])
              + interface["mux_bus_leakage_scenario_a"])
    ir = interface["adc_reference_current_scenario_a"]
    source = settings["excitation_v"]
    conductance = 1/(rc+rs) + 1/(2*rc)
    current_offset = ip*rs/(rc+rs) + im/2 + ir
    excitation = (source-excitation_path*current_offset)/(1+excitation_path*conductance)
    bridge_p = excitation*rs/(rc+rs) - ip*rc*rs/(rc+rs)
    bridge_m = excitation/2 - im*rc/2
    vp, vm = bridge_p-ip*path, bridge_m-im*path
    reference = excitation-ir*reference_path
    current = excitation*conductance+current_offset
    return dict(ratio=(vp-vm)/reference, nodes=np.stack([vp, vm], axis=-1),
                bridge_nodes=np.stack([bridge_p, bridge_m], axis=-1),
                excitation=excitation, reference=reference,
                source_current=current, source_power=source*current)


def ratio_response(raw, parameters, settings, interface, calibration=None):
    """Fixed zero-state uniform-strain calibration in Vdiff/Vref units."""
    _, rho, rc = shunt_design(parameters, settings)

    def response(strain):
        relative = 1+settings["gauge_factor"]*np.asarray(strain)[..., None, :]
        resistance = (settings["sensor_resistance_ohm"]*rho*relative/
                      (rho+(1-rho)*relative)).sum(-1)+settings["lead_resistance_per_string_ohm"]
        return switched_bridge(resistance, rc, settings, interface)

    if calibration is None:
        zero = response(np.zeros(64))["ratio"]
        gain = response(np.full(64, 1e-25j))["ratio"].imag/1e-25
    else:
        zero, gain = calibration
    result = response(raw)
    layers = (result["ratio"]-zero)/gain
    return dict(result, layers=layers, readout=components(layers), zero=zero,
                gain=gain, completion=rc)


def adc_codes(ratio, pga_gain):
    values = np.rint(np.asarray(ratio)*pga_gain*2**23)
    if np.any(values < -2**23) or np.any(values >= 2**23):
        raise ValueError("ADC input exceeds the signed 24-bit range")
    return values.astype(np.int32)


def encode_snapshot(codes, node_id, reference_id, sequence):
    """One signed sample in each 12-byte EPC, with durable snapshot identity.

    Bytes: type, version, node uint16, reference uint16, sequence uint16,
    channel uint8, sample int24. Multi-byte fields are big endian.
    The caller reserves a sequence before acquisition and changes reference
    identity before sequence reuse. Only completed snapshots are published.
    """
    if len(codes) != 12:
        raise ValueError("A snapshot must contain twelve channels")
    packets = []
    for channel, code in enumerate(codes):
        header = struct.pack(">BBHHHB", 0x53, 1, node_id, reference_id, sequence, channel)
        packets.append(header+int(code).to_bytes(3, "big", signed=True))
    return packets


def decode_snapshot(packets):
    """Reject incomplete, conflicting or mixed snapshots; allow retransmits."""
    identity, channels = None, {}
    for packet in packets:
        if len(packet) != 12:
            raise ValueError("Expected a 12-byte EPC")
        tag, version, node, reference, sequence, channel = struct.unpack(">BBHHHB", packet[:9])
        if (tag, version) != (0x53, 1) or channel >= 12:
            raise ValueError("Unsupported EPC format or channel")
        key = (node, reference, sequence)
        if identity is not None and identity != key:
            raise ValueError("Mixed snapshot identities")
        identity = key
        value = int.from_bytes(packet[9:], "big", signed=True)
        if channel in channels and channels[channel] != value:
            raise ValueError("Conflicting channel retransmission")
        channels[channel] = value
    if len(channels) != 12:
        raise ValueError("Incomplete snapshot")
    return identity, np.array([channels[i] for i in range(12)], dtype=np.int32)


def scan_budget(circuit, settings, interface):
    """Acquisition subtotal; RF, controller and regulator overhead stay explicit."""
    adc = settings["digital_alternative"]
    s = interface
    n = 12
    iadc = adc["analog_current_a"]+adc["digital_current_a"]+s["adc_external_reference_extra_current_a"]
    imux = s["mux_count"]*s["mux_quiescent_current_a"]
    supply_power = adc["supply_v"]*iadc+s["mux_supply_v"]*imux
    clock = adc["conversion_clock_hz"]
    tolerance = s["adc_internal_clock_fractional_tolerance"]
    rpath = circuit["completion"].max()/2+s["mux_on_resistance_ohm"]+s["signal_series_filter_ohm"]
    bus_cap = s["mux_on_capacitance_f"]+2*s["mux_off_drain_capacitance_f"]
    line_cap = s["signal_common_capacitance_f"]+s["cable_capacitance_per_signal_f"]+bus_cap
    # Matched-pair small-signal differential mode; excludes gauge thermal drift.
    tau = rpath*(line_cap+2*s["signal_differential_capacitance_f"])
    reference_tau = (2*s["mux_on_resistance_ohm"]+s["excitation_series_ohm"]+
                     s["reference_series_ohm"])*(s["reference_capacitance_f"]+bus_cap)
    # Upper charging scenario: every switched capacitor charged from zero to Vex.
    switched_cap = s["bridge_excitation_capacitance_f"]+s["reference_capacitance_f"]+2*line_cap+s["signal_differential_capacitance_f"]
    charge_switch = n*switched_cap*settings["excitation_v"]
    charge_cold_25 = settings["excitation_v"]*s["excitation_rail_capacitance_f"]
    charge_cold_33 = adc["supply_v"]*s["adc_mux_rail_capacitance_f"]
    vhi, vlo = settings["storage_voltage_high_v"], settings["storage_voltage_low_v"]
    results = []
    for rate, cycles in zip(adc["data_rates_sps"], adc["single_shot_conversion_cycles"]):
        conversion = np.array([cycles/(clock*(1+tolerance)),
            cycles/(clock*(1-tolerance))+adc["internal_oscillator_wakeup_max_s"]])
        on_time = conversion+s["settle_before_start_s"]+s["spi_start_and_read_bits_per_channel"]/s["spi_clock_hz"]
        slot = on_time+s["all_banks_disabled_s"]
        scan = n*slot+s["spi_configuration_bits"]/s["spi_clock_hz"]
        q25 = circuit["source_current"].sum(-1)[..., None]*on_time+charge_switch+charge_cold_25
        q33 = (iadc+imux)*scan+charge_cold_33
        energy = settings["excitation_v"]*q25+adc["supply_v"]*q33
        cap_linear = (q25+q33)/(vhi-vlo)
        cap_ideal = 2*energy/(vhi*vhi-vlo*vlo)

        def span(value, scale=1.):
            return [float(np.min(value)*scale), float(np.max(value)*scale)]

        results.append(dict(data_rate_sps=rate, scan_ms_range=(scan*1e3).tolist(),
            first_last_channel_start_separation_ms_range=((n-1)*slot*1e3).tolist(),
            acquisition_rail_energy_mj_range=span(energy, 1e3),
            ideal_conversion_storage_mf_range=span(cap_ideal, 1e3),
            linear_pass_storage_mf_range=span(cap_linear, 1e3),
            linear_pass_reservoir_energy_mj_range=span((q25+q33)*(vhi+vlo)/2, 1e3)))
    return dict(adc_and_mux_active_power_mw=supply_power*1e3,
        selected_source_power_mw_range=[float(circuit["source_power"].min()*1e3), float(circuit["source_power"].max()*1e3)],
        selected_excitation_current_ma_max=float(circuit["source_current"].max()*1e3),
        capacitor_connection_current_ma_bound=settings["excitation_v"]/
            min(s["excitation_series_ohm"], s["reference_series_ohm"])*1e3,
        differential_rc_time_constant_us=float(tau*1e6), reference_rc_time_constant_us=float(reference_tau*1e6),
        differential_step_remaining_fraction=float(np.exp(-s["settle_before_start_s"]/tau)),
        cold_start_rail_charge_microcoulomb=(charge_cold_25+charge_cold_33)*1e6,
        capacitor_charging_basis="Cold rail charge once per snapshot plus all switched capacitors charged from zero each slot; no controller or regulator startup charge included",
        scans=results)
