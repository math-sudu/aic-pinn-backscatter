"""Loaded resistor-string bridge and fixed-reference calibration."""

import numpy as np


def bridge_response(strain, count, settings):
    """Return raw amplifier voltage, source power and loaded input nodes.

    The sensor string is the lower arm of a quarter bridge. The three other
    arms match its zero-strain resistance, including fixed leads. Finite input
    impedances, bias currents and switch resistance load both bridge nodes.
    """
    strain = np.asarray(strain)
    rg = count * settings["sensor_resistance_ohm"]
    rc = rg + settings["lead_resistance_per_string_ohm"]
    rs = rg * (1 + settings["gauge_factor"] * strain) + settings["lead_resistance_per_string_ohm"]
    return loaded_bridge_response(rs, rc, settings)


def loaded_bridge_response(rs, rc, settings):
    """Solve the loaded quarter bridge for its actual sensor/completion arms."""
    rs, rc = np.asarray(rs), np.asarray(rc)
    if np.any(np.real(rs) <= 0):
        raise ValueError("Non-positive sensor-string resistance")
    if np.any(np.real(rc) <= 0):
        raise ValueError("Non-positive completion resistance")
    vex = settings["excitation_v"]
    vs = vex * rs / (rc + rs)
    vr = vex / 2
    rths = rc * rs / (rc + rs)
    rthr = rc / 2
    ron = settings["input_switch_resistance_ohm"]
    gs = 1 / (rths + ron)
    gr = 1 / (rthr + ron)
    gd = 1 / settings["input_differential_resistance_ohm"]
    gc = 1 / settings["input_common_mode_resistance_ohm"]
    bp, bm = settings["input_bias_a"]
    ap, am = gs + gd + gc, gr + gd + gc
    yp, ym = gs * vs - bp, gr * vr - bm
    determinant = ap * am - gd * gd
    vp = (yp * am + gd * ym) / determinant
    vm = (ym * ap + gd * yp) / determinant
    output = settings["amplifier_reference_v"] + settings["amplifier_gain"] * (vp - vm)
    bridge_p = vp + ron * (vs - vp) / (rths + ron)
    bridge_m = vm + ron * (vr - vm) / (rthr + ron)
    power = vex * ((vex - bridge_p) / rc + (vex - bridge_m) / rc)
    return output, power, np.stack([vp, vm], axis=-1)


def fixed_calibration(count, settings):
    zero = float(bridge_response(0.0, count, settings)[0])
    gain = float(np.imag(bridge_response(1e-25j, count, settings)[0]) / 1e-25)
    return zero, gain
