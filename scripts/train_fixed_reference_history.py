"""Train and save the force-driven material-history PINN in observable segments."""

import argparse
import json
import time

import numpy as np
import torch

from fixed_reference_history import (MaterialHistoryProblem, FixedReferenceNetwork, OUTPUT, ROOT,
                                     material_training_grid, parameter_indices)
from weighted_readout import FixedShuntReadout, uniform_parameters


def response_summary(values):
    difference = (values["circuit_increment"][:, 1:]-values["host_increment"][:, 1:])*1e6
    return dict(final_force_kn=float(values["force_n"][-1]/1000),
                central_host_final_microstrain=(values["host_increment"][:, -1, :, 1]*1e6).tolist(),
                circuit_host_rms_microstrain=float(np.sqrt(np.mean(difference**2))),
                maximum_history_micro_m=(values["maximum_history_m"][:, -1]*1e6).tolist(),
                maximum_damage=values["maximum_damage"][:, -1].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="physics")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-from")
    parser.add_argument("--path-coordinates", choices=["absolute", "reference_increments"])
    parser.add_argument("--training-block", choices=["all", "features", "output"])
    parser.add_argument("--width", type=int)
    parser.add_argument("--material-grid-order", type=int)
    parser.add_argument("--optimizer", choices=["lbfgs", "projected_newton", "gauss_newton", "regional_history"], default="lbfgs")
    parser.add_argument("--readout-from", help="Saved resistor selection used by regional-history training")
    parser.add_argument("--linear-tolerance", type=float, default=.02)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    record, checkpoint = OUTPUT/f"{args.name}.json", OUTPUT/f"{args.name}.pt"
    if args.resume and args.initialize_from:
        raise ValueError("Choose either resume or initialization from another checkpoint")
    source = OUTPUT/f"{args.initialize_from}.json" if args.initialize_from else record
    saved = json.loads(source.read_text()) if args.resume or args.initialize_from else None
    setup = saved["setup"] if saved else json.loads((ROOT/"inputs/fixed_reference_history.json").read_text())
    torch.manual_seed(setup["seed"])
    problem = MaterialHistoryProblem(setup)
    p = problem.physics
    source_coordinates = setup.get("path_coordinates", "reference_increments")
    network = FixedReferenceNetwork(problem.coordinates, problem.initial, setup["width"], source_coordinates)
    if saved:
        network.load_state_dict(torch.load(source.with_suffix(".pt"), weights_only=True)["network"])
    source_width = setup["width"]
    if args.width is not None and args.width != source_width:
        network.expand_features(args.width)
        setup["width"] = args.width
    network.reparameterize_path(args.path_coordinates or source_coordinates)
    setup["path_coordinates"] = network.path_coordinates
    setup["training_block"] = args.training_block or setup.get("training_block", "all")
    if args.optimizer != "lbfgs":
        if args.training_block == "features":
            raise ValueError("Projected history updates require an active output layer")
        setup["training_block"] = "output"
    if args.initialize_from:
        setup["initialization"] = (f"Load {source.with_suffix('.pt').name} and preserve all predictions "
                                    "before optimization. No response labels enter training.")
    setup["representation"] = (f"One common material-parameter network predicts {len(problem.initial)} complete states; "
        "path_coordinates records the temporal optimization coordinates. All four reflection sectors "
        "remain active. The reference state and all subsequent states enter the full physical history, "
        "and both host and nonlinear circuit observations subtract the fixed loaded reference.")
    if args.material_grid_order is not None:
        training = material_training_grid(args.material_grid_order)
        held = np.asarray(setup["query_parameters"])
        if np.any(np.all(np.isclose(training[:, None], held[None], rtol=0, atol=1e-12), axis=-1)):
            raise ValueError("The material training grid overlaps the held query set")
        setup["training_parameters"] = training.tolist()
        setup["material_grid_order"] = args.material_grid_order
        setup["training_sampling"] = "Tensor Chebyshev-Lobatto material nodes; equal physics weights"
    parameters = p.tensor(setup["training_parameters"])
    query = np.asarray(setup["training_parameters"]+setup["query_parameters"])
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    readout = FixedShuntReadout(uniform_parameters(), settings, p.tensor)
    regional, coefficients, installed = None, None, None
    if args.optimizer == "regional_history":
        previous_metric = setup.get("regional_history_training") if args.resume else None
        if previous_metric and not args.readout_from:
            design_source = previous_metric["design_source"]
            design_parameters = previous_metric["circuit_parameters"]
            coefficients = previous_metric["coefficients"]
        else:
            design_source = args.readout_from or args.initialize_from or args.name
            design = json.loads((OUTPUT/f"{design_source}_readout.json").read_text())
            design_parameters = design["designs"]["selected"]["parameters"]
        installed = FixedShuntReadout(design_parameters, settings, p.tensor)
        setup["regional_history_training"] = dict(design_source=design_source,
            circuit_parameters=design_parameters, outputs=["host", "circuit_minus_host"],
            targets="Local full-history Newton correction of fixed-reference outputs; refreshed per update",
            sampling="All physics-training materials; value and nominal-material change means; held inputs excluded")
        setup["initialization"] = (f"Load {source.with_suffix('.pt').name}; full physical residuals generate "
                                    "local regional targets. No external response labels enter training.")
    before = problem.evaluate(network, query, readout)
    if saved:
        with np.load(source.with_name(source.stem+"_query.npz")) as previous:
            indices = parameter_indices(query, previous["parameters"])
            for key in before:
                rebuilt = before[key] if key == "force_n" else before[key][indices]
                np.testing.assert_allclose(rebuilt, previous[key], rtol=1e-9, atol=1e-13)
    old_raw = network.raw(p.tensor(query)).detach()
    singular = network.condition_features(parameters)
    with torch.no_grad():
        conditioning_change = float((network.raw(p.tensor(query))-old_raw).abs().max())
    del old_raw
    hidden_before = {name: value.detach().clone() for name, value in network.layers[:-1].named_parameters()}
    features_before = network.features(parameters).detach()
    output_before = {name: value.detach().clone() for name, value in network.layers[-1].named_parameters()}
    if setup["training_block"] == "features":
        network.layers[-1].requires_grad_(False)
    elif setup["training_block"] == "output":
        network.layers[:-1].requires_grad_(False)
    active = [value for value in network.parameters() if value.requires_grad]
    print(json.dumps(dict(setup_seconds=time.perf_counter()-started, degrees=p.ndof,
        parities=problem.coordinates.parities, physical_force_kn=(problem.force[0]/1000).tolist(),
        network_parameters=sum(v.numel() for v in network.parameters()),
        training_block=setup["training_block"], active_parameters=sum(v.numel() for v in active),
        before=response_summary(before))), flush=True)
    steps = args.steps or setup["steps_per_segment"]
    optimizer = torch.optim.LBFGS(active, max_iter=steps,
        history_size=setup["lbfgs_history_size"], line_search_fn="strong_wolfe",
        tolerance_grad=1e-9, tolerance_change=1e-12) if args.optimizer == "lbfgs" else None
    evaluations, multiplier = [], None
    training_started = time.perf_counter()
    projected_iteration = 0

    def closure():
        nonlocal multiplier
        network.zero_grad(set_to_none=True)
        physics = []
        loss = problem.accumulate_loss(network, parameters, setup["batch_size"], physics)
        physical_loss = float(loss)
        terms = dict(physics=physical_loss)
        if regional is not None:
            goal, response_terms = regional.objective(network)
            goal.backward()
            loss = loss+goal.detach()
            terms.update(response_terms, weighted_regional=float(goal.detach()))
        if multiplier is None:
            multiplier = 1/float(loss)
        for value in active:
            value.grad.mul_(multiplier)
        row = dict(evaluation=len(evaluations)+1, objective=float(loss),
                   objective_terms=terms,
                   optimizer_iteration=(optimizer.state[active[0]].get("n_iter", 0)
                                        if optimizer is not None else projected_iteration),
                   seconds=time.perf_counter()-training_started, physics=physics)
        evaluations.append(row)
        critical = next(value for value in physics if value["parameters"] == [-1., -1.])
        print(json.dumps({key: value for key, value in row.items() if key != "physics"} | dict(
            weak_closure_peak_micro_m=critical["maximum_history_micro_m"][-1],
            weak_closure_damage=critical["maximum_damage"][-1])), flush=True)
        return loss*multiplier

    def save_progress(optimization):
        after = problem.evaluate(network, query, readout)
        with torch.no_grad():
            feature_update = dict(
                hidden_parameter_change_norm={name: float((value-hidden_before[name]).norm())
                                              for name, value in network.layers[:-1].named_parameters()},
                material_feature_change_rms=float((network.features(parameters)-features_before).square().mean().sqrt()),
                output_parameter_change_norm={name: float((value-output_before[name]).norm())
                                              for name, value in network.layers[-1].named_parameters()})
        if setup["training_block"] == "features" and any(feature_update["output_parameter_change_norm"].values()):
            raise ValueError("The feature-only block changed its fixed output coefficients")
        if setup["training_block"] == "output" and any(feature_update["hidden_parameter_change_norm"].values()):
            raise ValueError("The output-only block changed its fixed material features")
        segments = saved["segments"] if args.resume else []
        row = dict(segment=len(segments)+1, optimization=optimization,
            training_block=setup["training_block"], feature_update=feature_update,
            source_width=source_width, width=setup["width"],
            source_coordinates=source_coordinates, path_coordinates=network.path_coordinates,
            feature_singular_values=singular, conditioning_max_coordinate_change=conditioning_change,
            loss_evaluations=evaluations, training_seconds=time.perf_counter()-training_started,
            host_change_rms_microstrain=float(np.sqrt(np.mean((after["host_increment"]-before["host_increment"])**2))*1e6),
            before=response_summary(before), after=response_summary(after))
        OUTPUT.mkdir(exist_ok=True, parents=True)
        torch.save(dict(network=network.state_dict()), checkpoint)
        np.savez_compressed(OUTPUT/f"{args.name}_query.npz", **after)
        initialization = (dict(checkpoint=source.with_suffix(".pt").name, source_coordinates=source_coordinates)
                          if args.initialize_from else (saved.get("initialization") if saved else None))
        result = dict(setup=setup, name=args.name, initialization=initialization, device=str(p.device),
            segments=segments+[row], force_source=problem.boundary_path,
            parities=problem.coordinates.parities,
            interpretation="One shared network learns complete force-driven histories over two fixed material/contact parameters. Its reference state and all subsequent states are trained together using the same physical path residual. The saved base force path supplies the original common initialization; initialized runs name their checkpoint explicitly. Circuit outputs subtract the loaded reference after evaluating the exact nonlinear fixed-calibration circuit. Material observations and the other saved material equilibria are not training labels.")
        record.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(dict(saved=str(record), segment=row["segment"],
            accepted_updates=optimization["iterations"], stop_reason=optimization.get("stop_reason"))), flush=True)
        return row

    if optimizer is not None:
        optimizer.step(closure)
        state = next(iter(optimizer.state.values()))
        optimization = dict(method="lbfgs", iterations=state["n_iter"], evaluations=state["func_evals"],
                            loss_multiplier=multiplier, curvature_pairs=len(state.get("old_dirs", [])),
                            active_parameters=sum(value.numel() for value in active))
        row = save_progress(optimization)
    else:
        from history_optimization import gauss_newton_history_direction, projected_history_direction
        current = float(closure())
        updates = []
        def optimization_record(**extra):
            last = network.layers[-1]
            return dict(method=args.optimizer, iterations=len(updates), evaluations=len(evaluations),
                loss_multiplier=multiplier, updates=updates, workers=args.workers,
                active_parameters=last.weight.numel()+last.bias.numel(), **extra)

        for projected_iteration in range(1, steps+1):
            if args.optimizer == "regional_history":
                from regional_history_training import build_regional_metric
                regional = build_regional_metric(problem, network, parameters, installed,
                    evaluations[-1]["objective_terms"]["physics"], coefficients=coefficients, workers=args.workers)
                coefficients = regional.coefficients
                setup["regional_history_training"]["coefficients"] = coefficients
                current = float(closure())
            if args.optimizer in ("gauss_newton", "regional_history"):
                update, solve = gauss_newton_history_direction(problem, network, parameters,
                    args.linear_tolerance, regional=regional)
                direction_info = dict(linear_solve=solve)
                if regional is not None:
                    direction_info["regional_metric"] = regional.work
            else:
                update, directions = projected_history_direction(problem, network, parameters, args.workers)
                direction_info = dict(material_directions=directions)
            last = network.layers[-1]
            slope = float((last.weight.grad*update[:, :-1]).sum()+(last.bias.grad*update[:, -1]).sum())
            if args.optimizer in ("gauss_newton", "regional_history"):
                np.testing.assert_allclose(slope/multiplier, solve["directional_derivative"],
                                           rtol=1e-7, atol=1e-20)
            if slope >= 0:
                reason = ("non_descent_projected_direction" if args.optimizer == "projected_newton"
                          else "non_descent_gauss_newton_direction")
                save_progress(optimization_record(stop_reason=reason,
                    rejected_direction=dict(scaled_directional_derivative=slope, **direction_info)))
                raise RuntimeError("The shared history direction is not a physics-loss descent direction" +
                                   (" with regional targets" if regional is not None else ""))
            original_weight, original_bias = last.weight.detach().clone(), last.bias.detach().clone()
            original_raw = network.raw(p.tensor(query)).detach()
            with torch.no_grad():
                features = network.features(p.tensor(query))@network.feature_transform
                predicted_change = (features@update.T).reshape(original_raw.shape)
            scale = 1.
            while scale >= 1e-10:
                with torch.no_grad():
                    last.weight.copy_(original_weight+scale*update[:, :-1])
                    last.bias.copy_(original_bias+scale*update[:, -1])
                trial = float(closure())
                if trial <= current+1e-4*scale*slope:
                    break
                scale *= .5
            else:
                with torch.no_grad():
                    last.weight.copy_(original_weight)
                    last.bias.copy_(original_bias)
                save_progress(optimization_record(stop_reason="line_search_failed"))
                raise RuntimeError("No physics-loss decrease along the shared projected history direction")
            with torch.no_grad():
                torch.testing.assert_close(network.raw(p.tensor(query)), original_raw+scale*predicted_change,
                                           rtol=1e-9, atol=1e-13)
            updates.append(dict(iteration=projected_iteration, scale=scale,
                scaled_objective_before=current, scaled_objective_after=trial,
                scaled_directional_derivative=slope, **direction_info))
            current = trial
            optimization = optimization_record()
            row = save_progress(optimization)
    del optimizer
    print(json.dumps(dict(saved=str(record), segment=row["segment"], optimization=optimization,
                          after=row["after"], elapsed_seconds=time.perf_counter()-started)), flush=True)


if __name__ == "__main__":
    main()
