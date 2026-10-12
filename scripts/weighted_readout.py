"""Fixed passive shunts synthesize positive spatial weights before readout.

The same weight pattern is used at both depths. Right-hand regions mirror
left-hand regions, and each central pattern is even. No structural condition
is an input to the installed calibration. Mechanical strain transfer remains
the explicitly assumed 30 mm uniform kernel of the existing pilot.
"""

import numpy as np
import torch

from bridge_circuit import loaded_bridge_response
from functional_readout import loaded_bridge_voltage


COUNTS = (5, 6, 5)
PARAMETER_GROUPS = (slice(0, 5), slice(5, 8), slice(8, 13), slice(13, 16))


def weight_map():
    """Layer rows x individual gauges x shared circuit parameters."""
    mapping = np.zeros((12, 64, 16))
    cursor = 0
    for direction in range(2):
        for region, count in enumerate(COUNTS):
            for depth in range(2):
                row = (direction*3+region)*2+depth
                for j in range(count):
                    if region == 1:
                        parameter, multiplier = direction*8+5+min(j, 5-j), .5
                    else:
                        parameter = direction*8+(j if region == 0 else 4-j)
                        multiplier = 1.
                    mapping[row, cursor, parameter] = multiplier
                    cursor += 1
    return mapping


def uniform_parameters():
    return np.tile(np.r_[np.full(5, .2), np.full(3, 1/3)], 2)


def components(layers):
    value = layers.reshape(*layers.shape[:-1], 2, 3, 2)
    return np.stack([value.mean(axis=-1), (value[..., 1]-value[..., 0])/2], axis=-1)


def linear_features(raw):
    layers = np.einsum("...g,ogp->...op", raw, weight_map())
    # components expects the layer index at the end.
    return np.moveaxis(components(np.moveaxis(layers, -1, -2)), -4, -1)


def shunt_design(parameters, settings):
    """Choose the largest shunt retention at the prescribed normalized weights.

    rho = R_shunt / (R0 + R_shunt); rho=1 is an open shunt and
    rho=0 a gauge bypass. Zero-state normalized sensitivities are rho**2
    divided by their sum. Their common scale is fixed by max(rho)=1.
    """
    parameters = np.asarray(parameters)
    if parameters.shape != (16,) or np.any(parameters < 0):
        raise ValueError("Expected sixteen nonnegative circuit parameters")
    if any(not np.isclose(parameters[g].sum(), 1., atol=1e-8, rtol=0)
           for g in PARAMETER_GROUPS):
        raise ValueError("Each physical pattern must have unit weight")
    weights = np.einsum("ogp,p->og", weight_map(), parameters)
    rho = np.sqrt(weights/weights.max(axis=-1, keepdims=True))
    r0 = settings["sensor_resistance_ohm"]
    completion = r0*rho.sum(axis=-1)+settings["lead_resistance_per_string_ohm"]
    return weights, rho, completion


def weighted_response(raw, parameters, settings):
    """Exact shunted string, finite bridge loading and fixed zero-state gain."""
    raw = np.asarray(raw)
    if raw.shape[-1] != 64:
        raise ValueError("The installed layout has 64 ordered gauge strains")
    weights, rho, completion = shunt_design(parameters, settings)
    r0, gf = settings["sensor_resistance_ohm"], settings["gauge_factor"]

    def resistance(strain):
        relative = 1+gf*strain[..., None, :]
        if np.any(relative.real <= 0):
            raise ValueError("Non-positive active gauge resistance")
        return (r0*rho*relative/(rho+(1-rho)*relative)).sum(axis=-1)+settings["lead_resistance_per_string_ohm"]

    zero = loaded_bridge_response(completion, completion, settings)[0]
    # Uniform infinitesimal strain defines the calibration, independently of q.
    uniform = np.full(64, 1e-25j)
    gain = loaded_bridge_response(resistance(uniform), completion, settings)[0].imag/1e-25
    voltage, power, nodes = loaded_bridge_response(resistance(raw), completion, settings)
    layers = (voltage-zero)/gain
    return dict(readout=components(layers), layers=layers, voltage=voltage,
                power=power, nodes=nodes, zero=zero, gain=gain,
                weights=weights, rho=rho, completion=completion)


class FixedShuntReadout:
    """Exact differentiable state map with installed resistors and fixed gain.

    Circuit design is supplied explicitly; its values and zero-state calibration
    do not depend on the queried mechanical parameter or displacement.
    """
    def __init__(self, parameters, settings, tensor):
        self.settings = settings
        calibration = weighted_response(np.zeros(64), parameters, settings)
        self.rho = tensor(calibration["rho"])
        self.completion = tensor(calibration["completion"])
        self.zero = tensor(calibration["zero"])
        self.gain = tensor(calibration["gain"])

    def __call__(self, raw):
        s, rho = self.settings, self.rho
        relative = 1+s["gauge_factor"]*raw[..., None, :]
        if torch.any(relative <= 0):
            raise ValueError("Non-positive active gauge resistance")
        resistance = (s["sensor_resistance_ohm"]*rho*relative/
                      (rho+(1-rho)*relative)).sum(-1)+s["lead_resistance_per_string_ohm"]
        voltage = loaded_bridge_voltage(resistance, self.completion, s)
        layers = ((voltage-self.zero)/self.gain).reshape(*raw.shape[:-1], 2, 3, 2)
        return torch.stack([layers.mean(-1), (layers[..., 1]-layers[..., 0])/2], dim=-1)
