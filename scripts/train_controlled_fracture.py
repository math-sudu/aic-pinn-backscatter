"""Train the shared fractured path with exact actuator control and reactive forces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import time

import numpy as np
import torch

from controlled_fracture_history import ControlledFractureProblem, ExactControlNetwork, ROOT
from fixed_reference_history import OUTPUT, absolute_components
from weighted_readout import FixedShuntReadout, uniform_parameters


def initialize_worker(configuration):
    global worker_problem, worker_control
    torch.set_num_threads(2)
    worker_problem = ControlledFractureProblem(configuration)
    worker_control = ExactControlNetwork(torch.nn.Identity(), worker_problem)


def material_loss_gradient(problem, control, raw_values, parameter, path_coordinates):
    """One full material history; the returned gradient is in neural coordinates."""
    p = problem.physics
    raw = p.tensor(raw_values).requires_grad_(True)
    values = raw[None]
    if path_coordinates == "reference_increments":
        values = absolute_components(values)
    displacement = problem.coordinates.decode(values)*p.scale
    displacement = control.balance(displacement, p.tensor(parameter)[None])
    force = control.forces(displacement)
    residual = problem.equilibrium(displacement, force, p.tensor(parameter)[None])[0]
    loss = problem.coordinates.whiten(residual).square().mean()
    gradient = torch.autograd.grad(loss, raw)[0]
    return float(loss.detach()), gradient.numpy()


def worker_loss_gradient(payload):
    return material_loss_gradient(worker_problem, worker_control, *payload)


def accumulate_objective(problem, network, parameters, executor=None):
    if executor is None:
        per_material = []
        for eta in parameters:
            displacement, force = network(eta[None])
            residual = problem.equilibrium(displacement, force, eta[None])[0]
            loss = problem.coordinates.whiten(residual).square().mean()/len(parameters)
            loss.backward()
            per_material.append(float(loss.detach())*len(parameters))
    else:
        raw = network.raw(parameters)
        payloads = ((value, eta, network.displacement.path_coordinates)
                    for value, eta in zip(raw.detach().numpy(), parameters.numpy()))
        results = list(executor.map(worker_loss_gradient, payloads))
        per_material = [row[0] for row in results]
        raw.backward(problem.physics.tensor(np.stack([row[1] for row in results]))/len(parameters))
    return float(np.mean(per_material)), per_material


@torch.no_grad()
def evaluate(problem, network, parameters, readout):
    p = problem.physics
    volume = p.tensor(problem.fracture.volumes)
    body_nodes = [np.unique(p.mesh["tetra"][p.mesh["body"] == body]) for body in range(3)]
    rows = []
    for parameter in parameters:
        eta = p.tensor(parameter)[None]
        displacement, force = network(eta)
        residual, internal = problem.equilibrium(displacement, force, eta)
        q, circuit, raw = problem.base.responses(displacement, readout)
        reaction = p.mechanical_outputs(displacement)
        nodal_residual = residual[0].reshape(len(problem.initial), -1, 3)
        moment = torch.cross(p.tensor(p.mesh["xyz"])[None].expand_as(nodal_residual),
                             nodal_residual, dim=-1).sum(-2)
        body_force = torch.stack([nodal_residual[:, nodes].sum(-2) for nodes in body_nodes], dim=1)
        body_moment = torch.stack([torch.cross(p.tensor(p.mesh["xyz"])[nodes][None].expand_as(
            nodal_residual[:, nodes]), nodal_residual[:, nodes], dim=-1).sum(-2) for nodes in body_nodes], dim=1)
        nodal = displacement.reshape(1, len(problem.initial), -1, 3)
        interface_gap = torch.einsum("qa,bsfa->bsfq", p.shape,
            nodal[:, :, p.upper, 2]-nodal[:, :, p.lower, 2])[0]
        area = p.area[:, None]*p.surface_weights
        interface_damage = internal["interface_damage"][0].reshape_as(interface_gap)
        wall_area = p.wall_area[:, None]*p.surface_weights
        rows.append(dict(jack_force_n=force[0].numpy(),
            control_residual_m=problem.control(displacement, force)[0].numpy(),
            host_increment=q[0].numpy(), circuit_increment=circuit[0].numpy(),
            raw_gauge_strain=raw[0].numpy(),
            maximum_bulk_damage=internal["bulk_damage"][0].amax(-1).numpy(),
            damaged_volume_fraction=((internal["bulk_damage"][0] > 0)*volume).sum(-1).numpy()/float(volume.sum()),
            maximum_interface_damage=internal["interface_damage"][0].amax(-1).numpy(),
            interface_damaged_area_fraction=(((interface_damage > 0)*area).sum((-2, -1))/area.sum()).numpy(),
            interface_closed_area_fraction=(((interface_gap < 0)*area).sum((-2, -1))/area.sum()).numpy(),
            cone_contact_area_fraction=(((p.wall_gap(displacement)[0] < 0)*wall_area).sum(
                (-2, -1))/wall_area.sum()).numpy(),
            rod_force_n=reaction["rod_force_n"][0].numpy(),
            trestle_force_n=reaction["trestle_force_n"][0].numpy(),
            jack_face_m=reaction["jack_face_mean_m"][0].numpy(),
            local_relative_m=reaction["local_relative_m"][0].numpy(),
            force_balance_n=nodal_residual.sum(-2).numpy(),
            body_force_balance_n=body_force.numpy(),
            body_moment_balance_nm=body_moment.numpy(),
            moment_balance_nm=moment.numpy(),
            mechanical_loss_by_step=problem.coordinates.whiten(residual)[0].square().mean(-1).numpy(),
            vertical_balance_n=residual[0, :, 2::3].sum(-1).numpy()))
    return dict(parameters=np.asarray(parameters), **{
        key: np.stack([row[key] for row in rows]) for key in rows[0]})


def summary(values):
    nominal = int(np.flatnonzero(np.all(values["parameters"] == [1., -1.], axis=-1))[0])
    return dict(mechanical_loss=float(values["mechanical_loss_by_step"].mean()),
        maximum_later_force_balance_n=float(np.abs(values["force_balance_n"][:, 1:]).max()),
        maximum_later_moment_balance_nm=float(np.abs(values["moment_balance_nm"][:, 1:]).max()),
        maximum_later_body_resultant_kn=(np.linalg.norm(values["body_force_balance_n"][:, 1:],
            axis=-1).max(axis=(0, 1))/1e3).tolist(),
        maximum_later_body_moment_nm=np.linalg.norm(values["body_moment_balance_nm"][:, 1:],
            axis=-1).max(axis=(0, 1)).tolist(),
        maximum_control_residual_mm=float(np.abs(values["control_residual_m"]).max()*1e3),
        nominal_jack_force_kn=(values["jack_force_n"][nominal]/1e3).tolist(),
        nominal_vertical_balance_kn=(values["vertical_balance_n"][nominal]/1e3).tolist(),
        nominal_rod_sum_kn=(values["rod_force_n"][nominal].sum(-1)/1e3).tolist(),
        nominal_maximum_bulk_damage=values["maximum_bulk_damage"][nominal].tolist(),
        nominal_local_increment_mm=((values["local_relative_m"][nominal]
            -values["local_relative_m"][nominal, :1])*1e3).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="controlled_fracture")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-from")
    parser.add_argument("--balance-rigid", action="store_true")
    parser.add_argument("--balance-constituents", action="store_true")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--optimizer", choices=["lbfgs", "body_gauss_newton", "full_gauss_newton",
                                               "response_gauss_newton", "amplitude_gauss_newton",
                                               "opening_gauss_newton", "flexure_gauss_newton"], default="lbfgs")
    parser.add_argument("--linear-tolerance", type=float, default=.02)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    record = OUTPUT/f"{args.name}.json"
    if args.resume and args.initialize_from:
        parser.error("Use either --resume or --initialize-from")
    source_record = record if args.resume else (OUTPUT/f"{args.initialize_from}.json" if args.initialize_from else None)
    saved = json.loads(source_record.read_text()) if source_record else None
    if record.exists() and not args.resume:
        raise FileExistsError(f"Use --resume or a new run name: {record}")
    config = (saved["configuration"] if saved else
              json.loads((ROOT/"inputs/controlled_fracture_history.json").read_text()))
    if args.balance_rigid:
        config["rigid_balance"] = True
    if args.balance_constituents:
        config["constituent_balance"] = True
    if config.get("constituent_balance") and args.optimizer == "body_gauss_newton":
        parser.error("Constituent equilibrium already eliminates the body-motion update")
    problem = ControlledFractureProblem(config)
    p = problem.physics
    network = problem.load_network(exact_control=True, fixed_reference=bool(saved and saved.get("fixed_reference")))
    if saved:
        network.load_state_dict(torch.load(source_record.with_suffix(".pt"), weights_only=True)["network"])
    setup = problem.source["setup"]
    parameters = p.tensor(setup["training_parameters"])
    query = np.asarray(setup["training_parameters"]+setup["query_parameters"])
    readout = FixedShuntReadout(uniform_parameters(),
        json.loads((ROOT/"inputs/regional_readout.json").read_text()), p.tensor)
    before = evaluate(problem, network, query, readout)
    if args.resume:
        with np.load(record.with_name(record.stem+"_query.npz")) as previous:
            for key in previous.files:
                np.testing.assert_allclose(before[key], previous[key], rtol=1e-9, atol=1e-13)
    balance_initialization = saved.get("balance_initialization") if args.resume else None
    if args.initialize_from and (args.balance_rigid or args.balance_constituents):
        with np.load(source_record.with_name(source_record.stem+"_query.npz")) as previous:
            balance_initialization = dict(source=args.initialize_from,
                source_mechanical_loss=float(previous["mechanical_loss_by_step"].mean()),
                source_maximum_later_vertical_balance_n=float(np.abs(previous["vertical_balance_n"][:, 1:]).max()),
                source_maximum_later_moment_balance_nm=float(np.abs(previous["moment_balance_nm"][:, 1:]).max()),
                projected=summary(before),
                maximum_absolute_changes={key: float(np.abs(before[key]-previous[key]).max()) for key in
                    ("host_increment", "circuit_increment", "raw_gauge_strain", "maximum_bulk_damage",
                     "maximum_interface_damage", "local_relative_m")})
    # A causal reference uses the same saved feature coordinates as its fixed
    # coefficients. Free-reference runs may recondition the complete output.
    singular = (None if network.reference_coefficients is not None else
                network.displacement.condition_features(parameters))
    network.displacement.layers[:-1].requires_grad_(False)
    active = list(network.displacement.layers[-1].parameters())
    optimizer = (torch.optim.LBFGS(active, max_iter=args.steps,
        history_size=setup["lbfgs_history_size"], line_search_fn="strong_wolfe",
        tolerance_grad=1e-9, tolerance_change=1e-12) if args.optimizer == "lbfgs" else None)
    evaluations, multiplier = [], None
    training_started = time.perf_counter()
    active_count = sum(value.numel() for value in active)
    if args.optimizer != "lbfgs":
        width = {"body_gauss_newton": 12, "amplitude_gauss_newton": 1,
                 "opening_gauss_newton": 8, "flexure_gauss_newton": 8}.get(args.optimizer, network.displacement.shape[-1])
        active_count = (len(problem.initial)-1)*width*(network.displacement.layers[-1].in_features+1)
    print(json.dumps(dict(event="training_started", setup_seconds=training_started-started,
        material_conditions=len(parameters), physical_steps=len(problem.initial),
        active_parameters=active_count,
        before=summary(before))), flush=True)

    def closure():
        nonlocal multiplier
        optimizer.zero_grad(set_to_none=True)
        total, per_material = accumulate_objective(problem, network, parameters, executor)
        if multiplier is None:
            multiplier = 1/total
        for value in active:
            value.grad.mul_(multiplier)
        row = dict(evaluation=len(evaluations)+1, objective=total,
            objective_by_material=per_material,
            optimizer_iteration=optimizer.state[active[0]].get("n_iter", 0),
            seconds=time.perf_counter()-training_started)
        evaluations.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "objective_by_material"}), flush=True)
        return p.tensor(total*multiplier)

    if args.optimizer == "lbfgs":
        pool = (ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                   initargs=(config,)) if args.workers > 1 else nullcontext())
        with pool as executor:
            optimizer.step(closure)
        iterations = optimizer.state[active[0]]["n_iter"]
        updates = None
    else:
        if network.reference_coefficients is None or (network.rigid_balance is None and network.constituent_balance is None):
            raise ValueError("History updates require the prepared reference and reactive balance")
        if args.optimizer == "body_gauss_newton":
            from body_history_optimization import BodyMotionBasis, body_history_direction
            basis = BodyMotionBasis(problem)
            direction = lambda: body_history_direction(problem, network, parameters, basis, args.linear_tolerance)
            fold_update = basis.fold_update
            objective_key = "body_objective"
        else:
            from controlled_history_optimization import SharedControlledLinearization, controlled_history_direction
            if args.optimizer in ("opening_gauss_newton", "flexure_gauss_newton"):
                from regional_opening_basis import opening_history_direction
                direction = lambda: opening_history_direction(problem, network, parameters, args.linear_tolerance,
                    depth_kinematics="flexure" if args.optimizer == "flexure_gauss_newton" else "radial")
            elif args.optimizer in ("response_gauss_newton", "amplitude_gauss_newton"):
                from response_history_optimization import response_history_direction
                direction = lambda: response_history_direction(problem, network, parameters,
                    path_amplitudes=args.optimizer == "amplitude_gauss_newton", workers=args.workers)
            else:
                direction = lambda: controlled_history_direction(problem, network, parameters, args.linear_tolerance)
            fold_update = SharedControlledLinearization.fold_update
            objective_key = "objective"
        updates = []
        for iteration in range(1, args.steps+1):
            coefficients, linear, details = direction()
            if details["directional_derivative"] >= 0:
                raise RuntimeError("History Gauss-Newton direction is not a descent direction")
            scale = 1.
            while scale > np.finfo(float).eps:
                objective = linear.objective(scale*coefficients)
                evaluations.append(dict(iteration=iteration, scale=scale, **{objective_key: objective},
                                        seconds=time.perf_counter()-training_started))
                print(json.dumps(evaluations[-1]), flush=True)
                if objective <= details["objective_before"]+1e-4*scale*details["directional_derivative"]:
                    break
                scale *= .5
            else:
                raise RuntimeError("No decrease along the shared history direction")
            with torch.no_grad():
                update = fold_update(p.tensor(scale*coefficients))
                active[0].add_(update[:, :-1])
                active[1].add_(update[:, -1])
            updates.append(dict(iteration=iteration, scale=scale, objective_after=objective, **details))
            print(json.dumps(dict(history_update=updates[-1],
                                  seconds=time.perf_counter()-training_started)), flush=True)
            del linear
        iterations = len(updates)
    after = evaluate(problem, network, query, readout) if iterations else before
    segments = saved["segments"] if args.resume else []
    segment = dict(segment=len(segments)+1, method=args.optimizer,
        training_block=("body_motion" if args.optimizer == "body_gauss_newton" else "output"),
        workers=(args.workers if args.optimizer in ("lbfgs", "response_gauss_newton",
                                                   "amplitude_gauss_newton") else 1),
        iterations=iterations, evaluations=evaluations, loss_multiplier=multiplier,
        body_updates=(updates if args.optimizer == "body_gauss_newton" else None),
        field_updates=(updates if args.optimizer in ("full_gauss_newton", "response_gauss_newton",
                                                     "amplitude_gauss_newton", "opening_gauss_newton",
                                                     "flexure_gauss_newton") else None),
        feature_singular_values=singular,
        seconds=time.perf_counter()-training_started, before=summary(before), after=summary(after))
    torch.save(dict(network=network.state_dict()), record.with_suffix(".pt"))
    np.savez_compressed(record.with_name(record.stem+"_query.npz"), **after)
    result = dict(name=args.name, configuration=config, source_setup=setup,
        initialization=(saved.get("initialization") if args.resume else args.initialize_from),
        balance_initialization=balance_initialization,
        fixed_reference=network.reference_coefficients is not None,
        reference_preparation=saved.get("reference_preparation") if saved else None,
        control="Exact load elimination with positive series compliance; the learned reference remains coupled",
        training=("Shared body-motion equilibrium block with actual interface/contact history; equal material weights"
                  if args.optimizer == "body_gauss_newton" else
                  "Complete reactive force residual with nonlocal bulk and interface history; equal material weights"),
        segments=segments+[segment], elapsed_seconds=time.perf_counter()-started)
    record.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(saved=str(record), iterations=iterations, after=segment["after"])), flush=True)


if __name__ == "__main__":
    main()
