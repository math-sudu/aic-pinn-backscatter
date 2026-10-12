"""Follow a finite Newton correction in the saved controlled neural field.

This solves the selected acquisition with its preceding physical history kept.
It distinguishes a finite equilibrium correction from the earlier single
least-squares directions. The field is a diagnosis, not a shared trained model.
"""

import argparse
import json
import time

import numpy as np
from scipy.optimize import brentq
import torch

from analyze_postpeak_deformation import responses
from controlled_branch_solver import CurrentControlledTangent
from controlled_fracture_history import ControlledFractureProblem, ROOT
from fixed_reference_history import OUTPUT
from spatial_fracture import elastic_stress


@torch.no_grad()
def material_coordinates(problem, field, eta):
    young = problem.base.materials(eta[None])[:, 0]
    state = problem.fracture.stress_path(field[None], young)[1]
    return dict(maximum_current_equivalent_opening_mm=float(state["equivalent_opening_m"][0, -1].max()*1e3),
        maximum_history_strain=float(state["history"][0, -1].max()),
        first_opening_knee_mm=float(problem.fracture.law.opening_knots_m[1]*1e3))


@torch.no_grad()
def material_knee_seed(problem, network, baseline, eta):
    """A finite initializer fixed by the existing FRC law, not observations.

    Increase the checkpoint deformation until the largest averaged driving
    strain reaches the first sigma(w) knee. Optimizer trials do not add a
    fictitious damage event to the preceding physical history.
    """
    fracture, p = problem.fracture, problem.physics
    young = problem.base.materials(eta[None])[0, 0]
    strains = fracture.strains(baseline[None])[0]
    increment = strains[-1]-strains[0]
    target = float(fracture.law.stress_knots_pa[1]/young
                   +fracture.law.opening_knots_m[1]/fracture.softening_length)

    def maximum(scale):
        stress = elastic_stress(strains[0]+scale*increment, young, p.settings["bulk"]["poisson"])
        equivalent = torch.linalg.eigvalsh(stress)[:, -1].clamp_min(0)/young
        return float(torch.sparse.mm(fracture.weights, equivalent[:, None]).max())

    upper = 1.
    while maximum(upper) < target:
        upper *= 2
    scale = brentq(lambda value: maximum(value)-target, 0., upper)
    seed = torch.cat([baseline[:-1], baseline[:1]+scale*(baseline[-1:]-baseline[:1])])
    seed = network.balance(seed[None], eta[None])[0]
    return seed, dict(kind="first_material_knee", deformation_scale=scale,
        target_nonlocal_strain=target, opening_knee_mm=float(fracture.law.opening_knots_m[1]*1e3),
        role="Finite initialization only; the original preceding physical history is retained")


def solve_stage(problem, network, baseline, eta, *, reference_field, iterations,
                save_progress, linear_tolerance=.02, rebalance=None, direct=False,
                reference_objective=None, check_tangent=True):
    p = problem.physics
    nstep = len(baseline)
    current = baseline.clone()
    rebalance = network.balance if rebalance is None else rebalance

    def evaluate(field):
        force = torch.cat([field.new_tensor([problem.reference_force]),
            problem.reference_force+(problem.machine[1:nstep]
            -(field[1:]-field[:1])@p.jack_operator)/network.compliance])
        residual, state = problem.equilibrium(field[None], force[None], eta[None])
        return residual[0, -1], force, state

    @torch.no_grad()
    def objective(field):
        residual, force, state = evaluate(field)
        whitened = problem.coordinates.whiten(residual[None, None]).ravel()
        return float(whitened@whitened), residual, force, state

    initial_objective, _, initial_force, _ = objective(current)
    if reference_objective is None:
        reference_objective = objective(reference_field)[0]
    rows, checks = [], {}
    before = responses(problem, current[None], initial_force[None], eta[None])
    material_before = material_coordinates(problem, current, eta)

    def result(field, force, termination):
        with torch.no_grad():
            after = responses(problem, field[None], force[None], eta[None])
        return dict(before=before, after=after, iterations=rows, termination=termination,
                    solver_reference_objective=reference_objective, checks=checks.copy(),
                    material_coordinates=dict(before=material_before, after=material_coordinates(problem, field, eta)))

    save_progress(result(current, initial_force, "in_progress"), current, initial_force)
    for iteration in range(iterations):
        started = time.perf_counter()
        value, residual, force, state = objective(current)
        print(json.dumps(dict(iteration=iteration, event="assemble_current_tangent")), flush=True)
        tangent = CurrentControlledTangent(problem, current, eta)
        if iteration == 0 and check_tangent:
            # Independent action from the actual full-history autograd graph.
            rng = np.random.default_rng(6857)
            direction = p.tensor(rng.normal(size=p.ndof)*1e-6)
            point = current[-1].detach().requires_grad_(True)
            function = lambda last: evaluate(torch.cat([current[:-1], last[None]]))[0]
            image = torch.autograd.functional.jvp(function, point, direction)[1].detach().numpy()
            np.testing.assert_allclose(tangent@direction.numpy(), image, rtol=2e-8, atol=2e-6)
            probe = rng.normal(size=p.ndof)
            np.testing.assert_allclose(probe@(tangent@direction.numpy()),
                                       tangent.rmatvec(probe)@direction.numpy(), rtol=1e-10, atol=1e-7)
            checks.update(full_current_tangent=True, transpose=True)
        print(json.dumps(dict(iteration=iteration, event="solve_current_equation")), flush=True)
        direction, linear = tangent.solve(residual.numpy(), linear_tolerance, direct=direct)
        factor = 1.
        rejected_trials = []
        with torch.no_grad():
            # Preserve the causal constituent equilibrium at every trial. At a
            # balanced base its implicit first derivative agrees with the full
            # Newton equations; only their finite-step difference is corrected.
            for reduction in range(30):
                candidate = torch.cat([current[:-1], current[-1:]+factor*p.tensor(direction)[None]])
                try:
                    candidate = rebalance(candidate[None], eta[None])[0]
                except torch.linalg.LinAlgError:
                    rejected_trials.append(dict(scale=factor, reason="singular_constituent_tangent"))
                    factor *= .5
                    continue
                except RuntimeError as error:
                    if not str(error).startswith("Constituent equilibrium"):
                        raise
                    rejected_trials.append(dict(scale=factor, reason=str(error)))
                    factor *= .5
                    continue
                next_value, next_residual, next_force, next_state = objective(candidate)
                if next_value <= value*(1-1e-4*factor):
                    break
                factor *= .5
            else:
                report = result(current, force, "line_search_stalled")
                report["rejected_trials"] = rejected_trials
                return report, current, force
            np.testing.assert_allclose(candidate[:-1].numpy(), baseline[:-1].numpy(), rtol=0, atol=2e-13)
            local = ((candidate[-1]-candidate[0])@p.relative_operator.T).mean()*1e3
            rod = p.anchor_loads(candidate[-1]).sum()/1e3
            row = dict(iteration=iteration+1, accepted_scale=factor, linear=linear,
                predicted_force_kn=float(next_force[-1]/1e3), local_mean_increment_mm=float(local),
                rod_total_kn=float(rod), maximum_bulk_damage=float(next_state["bulk_damage"][0, -1].max()),
                objective=next_value, objective_fraction_of_initial=next_value/initial_objective,
                seconds=time.perf_counter()-started)
            row["objective_fraction_of_shared_reference"] = next_value/reference_objective
            row["rejected_trials"] = rejected_trials
            rows.append(row)
            current = candidate
        print(json.dumps(row), flush=True)
        save_progress(result(current, next_force, "in_progress"), current, next_force)
        # This is an equation-solver stopping setting, not an empirical
        # response criterion. Retain the same 0.02 scale as the linear solve.
        if next_value <= linear_tolerance**2*reference_objective:
            termination = "residual_reduced"
            break
    else:
        termination = "iteration_limit"
    with torch.no_grad():
        _, final_force, _ = evaluate(current)
    checks.update(earlier_fields_preserved=True, exact_control=True,
                  constituent_force_balance=True, strip_endpoint_integral=True)
    return result(current, final_force, termination), current, final_force


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="controlled_fracture_flexure")
    parser.add_argument("--stage", default="falling_300kn")
    parser.add_argument("--seed", choices=("shared", "knee"), default="shared")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--output")
    parser.add_argument("--rebuild-responses", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    record = OUTPUT/f"{args.name}.json"
    saved = json.loads(record.read_text())
    problem = ControlledFractureProblem(saved["configuration"])
    network = problem.load_network(exact_control=True, fixed_reference=saved["fixed_reference"])
    network.load_state_dict(torch.load(record.with_suffix(".pt"), weights_only=True)["network"])
    eta = problem.physics.tensor([1., -1.])
    step = list(problem.boundary["stage_labels"]).index(args.stage)
    if step == 0:
        parser.error("The prepared reference is fixed; select a later acquisition")
    with torch.no_grad():
        original, force = network(eta[None])
    output = ROOT/(args.output or ("results/ideation/B2_postpeak_branch"
        +("_knee" if args.seed == "knee" else "")+".json"))
    if args.rebuild_responses:
        report = json.loads(output.read_text())
        arrays = dict(np.load(output.with_suffix(".npz")))
        if report["checkpoint"] != record.relative_to(ROOT).as_posix() or report["stage"] != args.stage:
            raise ValueError("The saved branch belongs to another checkpoint or acquisition")
        field, load = problem.physics.tensor(arrays["displacement_m"]), problem.physics.tensor(arrays["force_n"])
        np.testing.assert_allclose(field[:-1], original[0, :step], rtol=0, atol=2e-13)
        with torch.no_grad():
            rebuilt = responses(problem, field[None], load[None], eta[None])
        for key in rebuilt:
            np.testing.assert_allclose(rebuilt[key], report["after"][key], rtol=1e-9, atol=1e-9)
        material = dict(before=material_coordinates(problem, problem.physics.tensor(arrays["initial_displacement_m"]), eta),
                        after=material_coordinates(problem, field, eta))
        if "material_coordinates" in report:
            for stage in material:
                for key in material[stage]:
                    np.testing.assert_allclose(material[stage][key], report["material_coordinates"][stage][key],
                                               rtol=1e-9, atol=1e-13)
        report["material_coordinates"] = material
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(dict(saved=str(output), reconstructed=True, after=rebuilt)), flush=True)
        return
    baseline = original[0, :step+1]
    previous = None
    if args.resume:
        previous = json.loads(output.read_text())
        if previous["checkpoint"] != record.relative_to(ROOT).as_posix() or previous["stage"] != args.stage:
            raise ValueError("The saved branch belongs to another checkpoint or acquisition")
        with np.load(output.with_suffix(".npz")) as arrays:
            baseline = problem.physics.tensor(arrays["displacement_m"].copy())
            initial_field = arrays["initial_displacement_m"].copy()
        np.testing.assert_allclose(baseline[:-1], original[0, :step], rtol=0, atol=2e-13)
        seed_info = previous["seed"]
    elif args.seed == "knee":
        baseline, seed_info = material_knee_seed(problem, network, baseline, eta)
    else:
        seed_info = dict(kind="shared_checkpoint")
    if previous is None:
        initial_field = baseline.numpy()

    def persist(report, field, load):
        report = dict(report)
        if previous is not None:
            offset = len(previous["iterations"])
            report["iterations"] = previous["iterations"]+[
                dict(row, iteration=offset+row["iteration"]) for row in report["iterations"]]
            report["before"] = previous["before"]
            report["material_coordinates"] = dict(report["material_coordinates"],
                before=previous["material_coordinates"]["before"])
        run_seconds = ((previous["run_elapsed_seconds"] if previous is not None else [])
                       +[time.perf_counter()-started])
        report.update(stage=args.stage, checkpoint=record.relative_to(ROOT).as_posix(),
            nominal_parameters=eta.tolist(), seed=seed_info,
            elapsed_seconds=sum(run_seconds), run_elapsed_seconds=run_seconds,
            role="Finite current-state equilibrium correction in the existing neural field; prior history retained, no shared weights or material inputs changed")
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        np.savez_compressed(output.with_suffix(".npz"), displacement_m=field.numpy(), force_n=load.numpy(),
                            initial_displacement_m=initial_field,
                            stage_labels=problem.boundary["stage_labels"][:step+1])
        return report

    report, field, load = solve_stage(problem, network, baseline, eta,
        reference_field=original[0, :step+1], iterations=args.iterations, save_progress=persist)
    report = persist(report, field, load)
    print(json.dumps(dict(saved=str(output), termination=report["termination"], after=report["after"],
                          seconds=report["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
