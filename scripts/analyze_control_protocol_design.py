"""Select fixed circuits on trained protocols and assess their actual response.

The physical evaluation fields and the intermediate reversal do not select
resistors. Every candidate uses the same installation and zero-state gain.
"""

import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
import torch

from analyze_weighted_neural_error import design_values
from controlled_fracture_history import ROOT
from fixed_reference_history import OUTPUT
from train_control_history import SOURCE, load_pilot, responses
from weighted_readout import PARAMETER_GROUPS, FixedShuntReadout, uniform_parameters, weighted_response


STATES = [0, 30, 31, 32]
NAMES = ("physics", "regional")


def compact(values):
    return dict(raw=np.take(values["raw_gauge_strain"], STATES, axis=1),
                host=np.take(values["host_microstrain"], STATES, axis=1).reshape(-1, 4, 2, 3, 2))


def circuit_values(data, parameters, settings):
    result = weighted_response(data["raw"], parameters, settings)
    values = (result["readout"]-result["readout"][:, :1])*1e6
    return values, result


def metrics(data, parameters, settings):
    values, _ = circuit_values(data, parameters, settings)
    error = values-data["host"]
    history = error[:, 3]-error[:, 1]
    terms = dict(response=float(np.mean(error[:, 1:]**2)), history=float(np.mean(history**2)))
    if len(values) == 2:
        terms["protocol_change"] = float(np.mean((history[0]-history[1])**2))
    return dict(**{name+"_rms_microstrain": float(np.sqrt(value)) for name, value in terms.items()},
                objective_microstrain2=sum(terms.values()),
                objective_root_microstrain=float(np.sqrt(sum(terms.values()))))


def select_design(data, settings):
    constraint = np.zeros((len(PARAMETER_GROUPS), 16))
    for index, group in enumerate(PARAMETER_GROUPS):
        constraint[index, group] = 1.
    start = uniform_parameters()
    scale = metrics(data, start, settings)["objective_microstrain2"]

    def objective(parameters):
        physical = parameters.copy()
        for group in PARAMETER_GROUPS:
            physical[group] /= physical[group].sum()
        return metrics(data, physical, settings)["objective_microstrain2"]/scale

    fit = minimize(objective, start, jac="3-point", method="SLSQP",
        bounds=Bounds(np.zeros(16), np.ones(16)),
        constraints=LinearConstraint(constraint, np.ones(4), np.ones(4)),
        options=dict(ftol=1e-10, maxiter=500))
    if not fit.success:
        raise RuntimeError(fit.message)
    np.testing.assert_allclose(constraint@fit.x, 1., rtol=0, atol=1e-8)
    if np.any(fit.x < 0) or fit.fun > 1.:
        raise ValueError("The selected passive circuit violates its constraints or worsens its starting objective")
    return fit.x, dict(iterations=int(fit.nit), evaluations=int(fit.nfev),
                      uniform_objective_microstrain2=scale,
                      selected_objective_microstrain2=fit.fun*scale)


@torch.no_grad()
def main():
    problem, network, eta, prefix, prefix_forces, _, _, _, fixed, _ = load_pilot()
    p = problem.physics
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    installed = json.loads((OUTPUT/"regional_history_readout.json").read_text())["designs"]["selected"]["parameters"]
    query_controls = p.tensor([[.6, .75, .6]])*1e-3
    fields, designs, fits = {}, {"installed": np.asarray(installed)}, {}
    for name in NAMES:
        checkpoint = ROOT/f"results/ideation/B2_control_protocol_{name}.json"
        record = json.loads(checkpoint.read_text())
        if record["status"] != "training_segment_completed":
            raise ValueError("Circuit selection requires a completed neural update")
        np.testing.assert_allclose(record["controls_mm"], [[.6, .8, .6], [.6, .7, .6]], rtol=0, atol=1e-12)
        data = compact(dict(np.load(checkpoint.with_suffix(".npz"))))
        network.load_state_dict(torch.load(checkpoint.with_suffix(".pt"), weights_only=True)["network"])
        query = responses(problem, network, eta, prefix, prefix_forces, query_controls, fixed)
        held = compact({key: value[None] for key, value in query.items()})
        fields[name] = dict(training=data, held=held)
        designs[name], fits[name] = select_design(data, settings)
        selected, _ = circuit_values(data, designs[name], settings)
        rebuilt = FixedShuntReadout(designs[name], settings, p.tensor)(p.tensor(data["raw"]))
        rebuilt = (rebuilt-rebuilt[:, :1]).numpy()*1e6
        np.testing.assert_allclose(selected, rebuilt, rtol=1e-8, atol=1e-7)

    physical = []
    sources = [SOURCE, ROOT/"results/ideation/B2_control_history_lower_peak.json",
               ROOT/"results/ideation/B2_control_history_mid_peak.json"]
    for path in sources:
        saved = np.load(path.with_suffix(".npz"))
        u = p.tensor(saved["displacement_m"][STATES])
        physical.append(dict(raw=(u@problem.base.raw_matrix.T).numpy(),
            host=((u-u[:1])@problem.base.functionals.q_matrix.T).numpy().reshape(4, 2, 3, 2)*1e6))
    fields["physical_evaluation"] = dict(
        training={key: np.stack([row[key] for row in physical[:2]]) for key in ("raw", "host")},
        held={key: physical[2][key][None] for key in ("raw", "host")})
    cross = {field: {group: {design: metrics(data, parameters, settings)
                           for design, parameters in designs.items()}
                     for group, data in groups.items()}
             for field, groups in fields.items()}
    changes = {}
    for group, data in fields["physical_evaluation"].items():
        values = {name: circuit_values(data, parameters, settings)[0] for name, parameters in designs.items()}
        changes[group] = dict(
            regional_minus_physics_output_rms_microstrain=float(np.sqrt(np.mean(
                (values["regional"][:, 1:]-values["physics"][:, 1:])**2))),
            history_by_design_microstrain={name: (value[:, 3]-value[:, 1]).tolist()
                                           for name, value in values.items()},
            regional_minus_physics_history_rms_microstrain=float(np.sqrt(np.mean((
                values["regional"][:, 3]-values["regional"][:, 1]
                -values["physics"][:, 3]+values["physics"][:, 1])**2))))
    report = dict(
        role="Fixed circuit selection after neural training, using only 0.7/0.8-mm neural protocols; 0.75-mm and physical evaluation fields are excluded from selection",
        objective="Circuit-minus-host squared values plus equal-control history differences and their high-minus-low reversal change, each block averaged independently; held evaluation reports values and its history difference",
        training_controls_mm=[[.6, .8, .6], [.6, .7, .6]], held_controls_mm=[.6, .75, .6],
        designs={name: design_values(value, settings) for name, value in designs.items()},
        selection=fits, cross_evaluation=cross, physical_output_change=changes)
    output = ROOT/"results/ideation/B2_control_protocol_design.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(cross["physical_evaluation"]), flush=True)


if __name__ == "__main__":
    main()
