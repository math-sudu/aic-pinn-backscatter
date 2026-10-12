"""Train a shared causal spatial head using the original history equilibrium.

Existing loading fields initialize a control interpolant. The unloading fields
are used only for response assessment, never in the physical training loss.
The supplied prefix and installed circuit stay fixed in this nominal-material
pilot. New protocols are queried through the same network weights.
"""

import argparse
import json
import time

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from control_history_network import ControlHistoryNetwork, LocalBalanceError
from controlled_fracture_history import ControlledFractureProblem, ROOT
from fixed_reference_history import OUTPUT
from weighted_readout import FixedShuntReadout


SOURCE = ROOT/"results/ideation/B2_unloading_history.json"
RESULT = ROOT/"results/ideation/B2_control_history_training.json"


def load_pilot(width=8):
    torch.set_num_threads(2)
    torch.manual_seed(6881)
    report = json.loads(SOURCE.read_text())
    problem = ControlledFractureProblem(report["configuration"])
    p = problem.physics
    data = np.load(SOURCE.with_suffix(".npz"))
    fields, forces = p.tensor(data["displacement_m"]), p.tensor(data["force_n"])
    network = ControlHistoryNetwork(problem, fields[29:32], fields[0], width=width)
    parameters = p.tensor(report["nominal_parameters"])[None]
    prefix, prefix_forces = fields[None, :30], forces[None, :30]
    controls = p.tensor([[.6, .8, .6, .8]])*1e-3
    design = json.loads((OUTPUT/"regional_history_readout.json").read_text())["designs"]["selected"]["parameters"]
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    readout = FixedShuntReadout(design, settings, p.tensor)
    return problem, network, parameters, prefix, prefix_forces, controls, fields, forces, readout, report


@torch.no_grad()
def responses(problem, network, parameters, prefix, prefix_forces, controls, readout, *, batched=False):
    if len(controls) != 1 and not batched:
        raise ValueError("Multiple protocols require batched response output")
    fields, forces = network(parameters, prefix, prefix_forces, controls)
    q = (fields-fields[:, :1])@problem.base.functionals.q_matrix.T
    raw = fields@problem.base.raw_matrix.T
    circuit = readout(raw).flatten(2)
    circuit = circuit-circuit[:, :1]
    start = prefix.shape[1]
    residual, state = problem.equilibrium(fields, forces, parameters)
    loss = problem.coordinates.whiten(residual[:, start:]).square().mean(dim=(1, 2))
    values = dict(displacement_m=fields.numpy(), force_n=forces.numpy(),
        machine_increment_m=network.machine_coordinates(fields, forces).numpy(),
        host_microstrain=q.numpy()*1e6, circuit_microstrain=circuit.numpy()*1e6,
        raw_gauge_strain=raw.numpy(), mechanical_loss=loss.numpy(),
        tail_bulk_damage_max=state["bulk_damage"][:, start:].max(-1).values.numpy(),
        tail_interface_damage_max=state["interface_damage"][:, start:].max(-1).values.numpy())
    return values if batched else {name: value[0] for name, value in values.items()}


def summary(values):
    if values["force_n"].ndim == 2:
        return dict(mechanical_loss=float(np.mean(values["mechanical_loss"])),
            protocols=[summary({name: value[index] for name, value in values.items()})
                       for index in range(len(values["force_n"]))])
    q, c = values["host_microstrain"], values["circuit_microstrain"]
    first, returned = 30, 32
    host, circuit = q[returned]-q[first], c[returned]-c[first]
    return dict(mechanical_loss=float(values["mechanical_loss"]),
        tail_force_kn=(values["force_n"][30:]/1e3).tolist(),
        unloading_host_microstrain=host.reshape(2, 3, 2).tolist(),
        unloading_circuit_microstrain=circuit.reshape(2, 3, 2).tolist(),
        unloading_host_rms_microstrain=float(np.sqrt(np.mean(host**2))),
        unloading_circuit_rms_microstrain=float(np.sqrt(np.mean(circuit**2))),
        unloading_discrepancy_rms_microstrain=float(np.sqrt(np.mean((circuit-host)**2))),
        gauge_abs_max_microstrain=float(np.abs(values["raw_gauge_strain"]).max()*1e6))


def lbfgs_direction(gradient, pairs, objective):
    direction = gradient.clone()
    coefficients = []
    for step, change in reversed(pairs):
        alpha = (step@direction)/(step@change)
        coefficients.append(alpha)
        direction -= alpha*change
    scale = ((pairs[-1][0]@pairs[-1][1])/(pairs[-1][1]@pairs[-1][1])
             if pairs else objective/(gradient@gradient))
    direction *= scale
    for (step, change), alpha in zip(pairs, reversed(coefficients)):
        beta = (change@direction)/(step@change)
        direction += step*(alpha-beta)
    return -direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--optimizer", choices=["lbfgs", "gauss_newton", "regional_history"], default="lbfgs")
    parser.add_argument("--output", default=str(RESULT.relative_to(ROOT)))
    parser.add_argument("--initialize-from", help="Existing checkpoint JSON to continue in a separate output")
    parser.add_argument("--protocols", help="JSON controls and history/protocol pairs for joint training")
    parser.add_argument("--correction-tolerance", type=float, default=.02)
    parser.add_argument("--linear-iterations", type=int, default=60)
    parser.add_argument("--linear-tolerance", type=float, default=.02)
    args = parser.parse_args()
    result_path = ROOT/args.output
    if result_path.exists() and not (args.resume or args.rebuild):
        parser.error("Use --resume or --rebuild for the existing checkpoint")
    if args.initialize_from and (args.resume or args.rebuild):
        parser.error("Use --initialize-from only for a new output")
    started = time.perf_counter()
    problem, network, eta, prefix, prefix_forces, controls, fields, forces, readout, source = load_pilot()
    saved = json.loads(result_path.read_text()) if args.resume or args.rebuild else None
    if saved and args.protocols:
        parser.error("Resume/rebuild uses the protocols stored in the checkpoint")
    protocol_spec = (saved.get("protocol_spec") if saved else
                     json.loads((ROOT/args.protocols).read_text()) if args.protocols else None)
    if protocol_spec:
        controls = problem.physics.tensor(protocol_spec["controls_mm"])*1e-3
        eta = eta.expand(len(controls), -1)
        prefix = prefix.expand(len(controls), -1, -1)
        prefix_forces = prefix_forces.expand(len(controls), -1)
    initialization = ROOT/args.initialize_from if args.initialize_from else None
    checkpoint = result_path if saved else initialization
    if checkpoint:
        network.load_state_dict(torch.load(checkpoint.with_suffix(".pt"), weights_only=True)["network"])
    def collect():
        return responses(problem, network, eta, prefix, prefix_forces, controls, readout,
                         batched=protocol_spec is not None)

    before = collect()
    if args.rebuild:
        data = np.load(result_path.with_suffix(".npz"))
        for name, value in before.items():
            np.testing.assert_allclose(value, data[name], rtol=1e-8, atol=1e-9)
        print(json.dumps(dict(rebuilt=summary(before))), flush=True)
        return
    normalization = float(np.mean(before["mechanical_loss"]))
    parameters = list(network.parameters())
    rows = list(saved["iterations"]) if saved else []
    pairs = []
    evaluations = 0
    regional = None
    coefficients = saved.get("regional_coefficients") if saved else None

    def evaluate():
        nonlocal evaluations
        network.zero_grad(set_to_none=True)
        predicted, predicted_forces = network(eta, prefix, prefix_forces, controls)
        residual = problem.equilibrium(predicted, predicted_forces, eta)[0][:, prefix.shape[1]:]
        loss = problem.coordinates.whiten(residual).square().mean()
        if regional is not None:
            loss = loss+regional.objective(predicted)
        loss = loss/normalization
        loss.backward()
        gradient = parameters_to_vector([parameter.grad for parameter in parameters]).detach()
        evaluations += 1
        print(json.dumps(dict(evaluation=evaluations, normalized_loss=float(loss.detach()),
                             elapsed_seconds=time.perf_counter()-started)), flush=True)
        return float(loss.detach()), gradient

    def save(values, status):
        report = dict(source_path=SOURCE.relative_to(ROOT).as_posix(), configuration=source["configuration"],
            material_parameters=eta[0].tolist(), prefix_states=prefix.shape[1],
            initialization_loading_indices=[29, 30, 31],
            initialization_role="Loading-field warm start only; no unloading labels in the loss",
            controls_mm=(controls*1e3).tolist() if protocol_spec else (controls[0]*1e3).tolist(),
            protocol_spec=protocol_spec, encoder_width=network.encoder.cell.hidden_size,
            input_role="Material coordinates, current local displacement and signed increment; no step or branch ID",
            prefix_role="Complete saved nominal-material physical history supplied unchanged",
            training_objective=("Full-domain force residual and physics-generated H=(host, circuit-host) values, equal-control history differences and configured protocol contrasts"
                if args.optimizer == "regional_history" else "Full-domain force residual over predicted tail states, with causal bulk/interface history"),
            initialization_checkpoint=(initialization.relative_to(ROOT).as_posix() if initialization
                else saved.get("initialization_checkpoint") if saved else None),
            regional_coefficients=coefficients,
            fixed_circuit=OUTPUT.relative_to(ROOT).as_posix()+"/regional_history_readout.json",
            before=saved["before"] if saved else summary(before), after=summary(values),
            iterations=rows, status=status,
            elapsed_seconds=(saved["elapsed_seconds"] if saved else 0)+time.perf_counter()-started)
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        torch.save(dict(network=network.state_dict()), result_path.with_suffix(".pt"))
        np.savez_compressed(result_path.with_suffix(".npz"), **values)

    current, gradient = evaluate()
    save(before, "initialized")
    for _ in range(args.steps):
        point = parameters_to_vector(parameters).detach().clone()
        linear_details = None
        if args.optimizer == "regional_history":
            from control_history_response_training import build_control_history_metric
            regional = build_control_history_metric(network, eta, prefix, prefix_forces, controls,
                readout, coefficients=coefficients, tolerance=args.correction_tolerance,
                history_pairs=protocol_spec["history_pairs"] if protocol_spec else ((0, 2),),
                protocol_pairs=protocol_spec["protocol_pairs"] if protocol_spec else ())
            coefficients = regional.coefficients
            current, gradient = evaluate()
        if args.optimizer in ("gauss_newton", "regional_history"):
            from control_history_optimization import control_history_direction
            update, linear_details = control_history_direction(network, eta, prefix, prefix_forces,
                controls, maxiter=args.linear_iterations, tolerance=args.linear_tolerance, regional=regional)
            shifts = [torch.zeros_like(parameter) for parameter in network.encoder.parameters()]
            shifts += [network.head.weight.new_tensor(update[:-1].T.copy()), network.head.bias.new_tensor(update[-1])]
            direction = parameters_to_vector(shifts)
        else:
            direction = lbfgs_direction(gradient, pairs, current)
        slope = float(gradient@direction)
        if not np.isfinite(slope) or slope >= 0:
            raise RuntimeError("The physical training direction is not a finite descent direction")
        alpha, rejected = 1., []
        for _ in range(30):
            vector_to_parameters(point+alpha*direction, parameters)
            try:
                candidate, next_gradient = evaluate()
            except LocalBalanceError as error:
                rejected.append(dict(scale=alpha, reason=str(error)))
            else:
                if np.isfinite(candidate) and candidate <= current+1e-4*alpha*slope:
                    break
                rejected.append(dict(scale=alpha, normalized_loss=candidate))
            alpha *= .5
        else:
            vector_to_parameters(point, parameters)
            save(collect(), "line_search_stopped")
            raise RuntimeError("No accepted physical training update; checkpoint remains at last accepted state")
        step, change = alpha*direction, next_gradient-gradient
        if float(step@change) > 0:
            pairs.append((step, change))
            pairs = pairs[-8:]
        current, gradient = candidate, next_gradient
        values = collect()
        rows.append(dict(index=len(rows)+1, method=args.optimizer, linearization=linear_details,
                         regional_metric=regional.work if regional is not None else None,
                         scale=alpha, rejected=rejected, **summary(values)))
        save(values, "training_segment_completed")
        print(json.dumps(dict(accepted_update=rows[-1])), flush=True)


if __name__ == "__main__":
    main()
