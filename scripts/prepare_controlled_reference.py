"""Train the known-load reference first, then hold it as the causal initial state."""

import argparse
import json
import time

import numpy as np
import torch

from controlled_fracture_history import ControlledFractureProblem, ROOT
from fixed_reference_history import FixedReferenceNetwork, OUTPUT
from train_controlled_fracture import evaluate, summary
from weighted_readout import FixedShuntReadout, uniform_parameters


@torch.no_grad()
def reference_observables(problem, network, parameters):
    rows = []
    for eta in parameters:
        state = network(eta[None])
        force = eta.new_tensor([[problem.reference_force]])
        residual, internal = problem.equilibrium(state, force, eta[None])
        nodal = residual[0, 0].reshape(-1, 3)
        rows.append(dict(loss=float(problem.coordinates.whiten(residual).square().mean()),
            vertical_balance_n=float(nodal[:, 2].sum()),
            moment_balance_nm=torch.cross(problem.physics.tensor(problem.physics.mesh["xyz"]),
                                         nodal, dim=-1).sum(0).tolist(),
            maximum_bulk_damage=float(internal["bulk_damage"].max()),
            maximum_interface_damage=float(internal["interface_damage"].max())))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initialize-from", default="controlled_fracture")
    parser.add_argument("--name", default="controlled_fracture_causal")
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    source = OUTPUT/f"{args.initialize_from}.json"
    destination = OUTPUT/f"{args.name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    saved = json.loads(source.read_text())
    problem = ControlledFractureProblem(saved["configuration"])
    p, setup = problem.physics, problem.source["setup"]
    original = problem.load_network(exact_control=True)
    # The original low-load learned reference supplies an intact starting guess.
    reference = FixedReferenceNetwork(problem.coordinates, problem.initial[:1], setup["width"], "absolute")
    for index in (0, 2):
        reference.layers[index].load_state_dict(original.displacement.layers[index].state_dict())
    with torch.no_grad():
        reference.feature_transform.copy_(original.displacement.feature_transform)
        count = reference.shape[-1]
        reference.layers[-1].weight.copy_(original.displacement.layers[-1].weight[:count])
        reference.layers[-1].bias.copy_(original.displacement.layers[-1].bias[:count])
    reference.layers[:-1].requires_grad_(False)
    parameters = p.tensor(setup["training_parameters"])
    query = p.tensor(setup["training_parameters"]+setup["query_parameters"])
    before = reference_observables(problem, reference, query)
    reference.condition_features(parameters)
    optimizer = torch.optim.LBFGS(reference.layers[-1].parameters(), max_iter=args.steps,
        history_size=setup["lbfgs_history_size"], line_search_fn="strong_wolfe",
        tolerance_grad=1e-9, tolerance_change=1e-12)
    evaluations, multiplier = [], None

    def closure():
        nonlocal multiplier
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for eta in parameters:
            state = reference(eta[None])
            residual = problem.equilibrium(state, eta.new_tensor([[problem.reference_force]]), eta[None])[0]
            loss = problem.coordinates.whiten(residual).square().mean()/len(parameters)
            loss.backward()
            total += float(loss.detach())
        if multiplier is None:
            multiplier = 1/total
        for value in reference.layers[-1].parameters():
            value.grad.mul_(multiplier)
        row = dict(evaluation=len(evaluations)+1, objective=total, seconds=time.perf_counter()-started)
        evaluations.append(row)
        print(json.dumps(row), flush=True)
        return p.tensor(total*multiplier)

    optimizer.step(closure)
    after = reference_observables(problem, reference, query)
    network = problem.load_network(exact_control=True)
    network.load_state_dict(torch.load(source.with_suffix(".pt"), weights_only=True)["network"])
    network.displacement.reparameterize_path("absolute")
    for actual, expected in zip(network.displacement.layers[:-1].parameters(), reference.layers[:-1].parameters()):
        torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        coefficient = torch.cat([reference.layers[-1].weight, reference.layers[-1].bias[:, None]], dim=-1)
        translated = torch.linalg.solve(network.displacement.feature_transform,
                                        reference.feature_transform@coefficient.T).T
        network.displacement.layers[-1].weight[:count].copy_(translated[:, :-1])
        network.displacement.layers[-1].bias[:count].copy_(translated[:, -1])
    network.fix_reference()
    with torch.no_grad():
        for eta in query:
            torch.testing.assert_close(network(eta[None])[0][:, :1], reference(eta[None]), atol=1e-13, rtol=1e-9)
    readout = FixedShuntReadout(uniform_parameters(),
        json.loads((ROOT/"inputs/regional_readout.json").read_text()), p.tensor)
    values = evaluate(problem, network, query.numpy(), readout)
    preparation = dict(source_high_load_checkpoint=source.with_suffix(".pt").relative_to(ROOT).as_posix(),
        initial_reference=problem.config["source_network"],
        method="Train reference equilibrium at the known jack force independently; retain its material function as the causal initial state",
        iterations=optimizer.state[next(iter(reference.layers[-1].parameters()))]["n_iter"],
        evaluations=evaluations, parameters=query.tolist(), before=before, after=after,
        prepared_path=summary(values))
    torch.save(dict(network=network.state_dict()), destination.with_suffix(".pt"))
    np.savez_compressed(destination.with_name(destination.stem+"_query.npz"), **values)
    result = dict(name=args.name, configuration=problem.config, source_setup=setup,
        fixed_reference=True, reference_preparation=preparation, segments=[],
        control="Exact series-compliance load elimination relative to the independently trained reference",
        training="Causal initial state prepared; later displacement outputs retain the previous trial and require reactive-history continuation",
        elapsed_seconds=time.perf_counter()-started)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(saved=str(destination),
        reference_maximum_vertical_imbalance_n=max(abs(row["vertical_balance_n"]) for row in after),
        reference_maximum_bulk_damage=max(row["maximum_bulk_damage"] for row in after),
        seconds=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
