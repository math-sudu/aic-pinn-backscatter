"""Test broad three-body deformation in the saved nonlinear physical history.

Each body has every affine strain and quadratic displacement, with rigid
motions supplied by the existing constituent layer. This is a field-direction
diagnostic; measured displacements and forces never determine its coefficients.
"""

import argparse
import json
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
import torch

from analyze_regional_transfer import body_resultants, resolve_terms
from analyze_regional_redistribution import EDGES, regional_components, sample_displacement
from controlled_fracture_history import ControlledFractureProblem, ROOT
from controlled_history_optimization import MaterialControlledHistory
from fixed_reference_history import OUTPUT


def polynomial_fields(mesh):
    """Twenty-four non-rigid polynomial directions per body, per metre.

    Six symmetric affine fields span all constant strains. Eighteen quadratic
    vector monomials complete the vector P2 space modulo its six rigid modes.
    One length per body preserves the symmetry of affine shear fields.
    """
    fields, names = [], []
    for body, body_name in enumerate(("top", "substrate", "cone")):
        nodes = np.unique(mesh["tetra"][mesh["body"] == body])
        xyz = mesh["xyz"][nodes]
        centre = (xyz.max(0)+xyz.min(0))/2
        length = np.ptp(xyz, axis=0).max()/2
        position = (xyz-centre)/length
        for i in range(3):
            for j in range(i, 3):
                field = np.zeros_like(mesh["xyz"])
                field[nodes, i] = position[:, j]
                field[nodes, j] = position[:, i]
                fields.append(field.ravel())
                names.append(f"{body_name}_strain_{'xyz'[i]}{'xyz'[j]}")
        for component in range(3):
            for i in range(3):
                for j in range(i, 3):
                    field = np.zeros_like(mesh["xyz"])
                    field[nodes, component] = position[:, i]*position[:, j]
                    fields.append(field.ravel())
                    names.append(f"{body_name}_u{'xyz'[component]}_{'xyz'[i]}{'xyz'[j]}")
    return np.asarray(fields), names


def check_directional_derivative(finite, image):
    """Check one physical direction without per-row relative tests at zeros."""
    difference = np.linalg.norm(np.asarray(finite)-np.asarray(image))
    scale = np.linalg.norm(image)
    if difference > 2e-4*scale:
        raise AssertionError(f"The physical directional derivative differs by {difference/scale}")


def responses(problem, displacement, force, eta):
    p = problem.physics
    outputs = p.mechanical_outputs(displacement)
    host = p.observations(displacement-displacement[:, :1])[0]
    terms, _, state = resolve_terms(problem, displacement, force, eta)
    top_to_bodies = body_resultants(p, terms["interface"])[0, -1]
    increment = (displacement[0, -1]-displacement[0, 0]).numpy()
    extensions = []
    for direction in (0, 1):
        layers = []
        for depth in p.settings["observations"]["depths_m"]:
            points = np.zeros((len(EDGES), 3))
            points[:, direction], points[:, 2] = EDGES, depth
            sampled = sample_displacement(p, increment, 0, points)
            layers.append(np.diff(sampled[:, direction])*1e3)
        extensions.append(layers)
    extensions = np.asarray(extensions)
    np.testing.assert_allclose(regional_components(extensions), host[0, -1].numpy()*1e6,
                               rtol=1e-7, atol=1e-6)
    control = ((displacement[:, 1:]-displacement[:, :1])@p.jack_operator
        +problem.config["apparatus"]["compliance_m_per_n"]*(force[:, 1:]-problem.reference_force)
        -problem.machine[None, 1:displacement.shape[1]])
    np.testing.assert_allclose(control.numpy(), 0., rtol=0, atol=1e-16)
    body_force = body_resultants(p, sum(terms.values()))[:, 1:]
    np.testing.assert_allclose(body_force.numpy(), 0., rtol=0, atol=2e-4)
    return dict(predicted_force_kn=float(force[0, -1]/1e3),
        local_mean_increment_mm=float(((outputs["local_relative_m"][:, -1]
            -outputs["local_relative_m"][:, 0]).mean())*1e3),
        jack_increment_mm=float(((displacement[:, -1]-displacement[:, 0])@p.jack_operator)[0]*1e3),
        rod_total_kn=float(outputs["rod_force_n"][0, -1].sum()/1e3),
        cone_to_top_kn=float(top_to_bodies[2, 2]/1e3),
        host_microstrain=(host[0, -1].numpy()*1e6).tolist(),
        strip_extension_mm=extensions.tolist(),
        maximum_bulk_damage=float(state["damage"][0, -1].max()))


class TerminalDeformationLinearization(LinearOperator):
    """Actual balanced residual over every final-state neural spatial DOF."""

    def __init__(self, problem, network, eta, baseline):
        self.problem, self.network, self.eta, self.baseline = problem, network, eta, baseline
        self.point = problem.physics.tensor(problem.coordinates.encode_nominal(
            baseline[0, -1:].numpy()))[0].requires_grad_(True)
        self.value = self.function(self.point)
        self.dual = torch.zeros_like(self.value, requires_grad=True)
        self.pullback = torch.autograd.grad(self.value, self.point, self.dual, create_graph=True)[0]
        self.products = dict(forward=0, transpose=0)
        super().__init__(dtype=np.dtype(float), shape=(len(self.value), len(self.point)))

    def fields(self, point):
        p, network = self.problem.physics, self.network
        field = self.problem.coordinates.decode(point[None, None])*p.scale
        raw = torch.cat([self.baseline[:, :-1], field], dim=1)
        displacement = network.balance(raw, self.eta)
        nstep = displacement.shape[1]
        force = network.reference_force+(network.machine[None, 1:nstep]
            -(displacement[:, 1:]-displacement[:, :1])@p.jack_operator)/network.compliance
        return displacement, torch.cat([network.reference_force.expand(1, 1), force], dim=1)

    def function(self, point):
        displacement, force = self.fields(point)
        residual = self.problem.equilibrium(displacement, force, self.eta)[0]
        return self.problem.coordinates.whiten(residual)[0, -1]

    def _matvec(self, direction):
        self.products["forward"] += 1
        result = torch.autograd.grad(self.pullback, self.dual,
            self.point.new_tensor(np.asarray(direction).reshape(-1)), retain_graph=True)[0]
        print(json.dumps(dict(full_terminal_products=self.products)), flush=True)
        return result.detach().numpy()

    def _rmatvec(self, direction):
        self.products["transpose"] += 1
        return torch.autograd.grad(self.value, self.point,
            self.value.new_tensor(np.asarray(direction).reshape(-1)), retain_graph=True)[0].detach().numpy()


def full_field_trial(problem, network, baseline, *, tolerance=.02):
    p = problem.physics
    eta = p.tensor([[1., -1.]])
    linear = TerminalDeformationLinearization(problem, network, eta, baseline)
    before = linear.value.detach().numpy()
    solved = lsmr(linear, -before, atol=tolerance, btol=tolerance)
    direction, stop, iterations, residual_norm, normal_norm, operator_norm, condition, norm = solved
    image = linear@direction
    derivative = 2*before@image
    if derivative >= 0:
        raise RuntimeError("The full field trial is not a physical residual descent direction")
    with torch.no_grad():
        factor = 1.
        initial_objective = float(before@before)
        while factor > np.finfo(float).eps:
            trial_point = linear.point+factor*p.tensor(direction)
            after = linear.function(trial_point).numpy()
            if after@after <= initial_objective+1e-4*factor*derivative:
                break
            factor *= .5
        else:
            raise RuntimeError("The full field trial line search failed")
        displacement, force = linear.fields(trial_point)
        np.testing.assert_allclose(displacement[:, :-1], baseline[:, :-1], rtol=0, atol=2e-13)
        h = 1e-4
        finite = (linear.function(linear.point+h*p.tensor(direction))
            -linear.function(linear.point-h*p.tensor(direction))).numpy()/(2*h)
        check_directional_derivative(finite, image)
        after_response = responses(problem, displacement, force, eta)
    # Check the dual action using a different deterministic residual direction.
    probe = np.random.default_rng(6856).normal(size=len(before))
    np.testing.assert_allclose(probe@image, linear.rmatvec(probe)@direction, rtol=1e-9, atol=1e-12)
    report = dict(role="One complete spatial neural-field direction at the final acquisition; no shared weights changed",
        spatial_dofs=len(direction), linear_tolerance=tolerance, stop_code=int(stop),
        iterations=int(iterations), accepted_scale=factor, after=after_response,
        objective_before=initial_objective, linearized_objective_after=float(np.sum((before+image)**2)),
        objective_after=float(after@after), residual_norm=float(residual_norm),
        normal_residual_norm=float(normal_norm), condition_estimate=float(condition),
        products=linear.products, checks=dict(earlier_fields_preserved=True, mixed_tangent=True, transpose=True))
    arrays = dict(full_trial_displacement_m=displacement[0].numpy(), full_trial_force_n=force[0].numpy(),
                  full_trial_raw_direction=direction)
    return report, arrays


@torch.no_grad()
def rebuild_responses(problem, report, arrays):
    """Rebuild scientific outputs from saved physical fields and their history."""
    p = problem.physics
    eta = p.tensor([[1., -1.]])
    original = p.tensor(arrays["original_displacement_m"])[None]
    # The baseline load follows its saved exact control relation.
    force = problem.reference_force+(problem.machine[None, 1:original.shape[1]]
        -(original[:, 1:]-original[:, :1])@p.jack_operator)/problem.config["apparatus"]["compliance_m_per_n"]
    original_force = torch.cat([p.tensor([[problem.reference_force]]), force], dim=1)
    report["before"] = responses(problem, original, original_force, eta)
    for key, prefix in ((None, "trial"), ("full_field_trial", "full_trial")):
        if key is not None and key not in report:
            continue
        field = p.tensor(arrays[f"{prefix}_displacement_m"])[None]
        force = p.tensor(arrays[f"{prefix}_force_n"])[None]
        (report if key is None else report[key])["after"] = responses(problem, field, force, eta)
    if "full_history_trial" in report:
        field = p.tensor(arrays["history_trial_displacement_m"])[None]
        force = p.tensor(arrays["history_trial_force_n"])[None]
        labels = list(arrays["stage_labels"])
        for row in report["full_history_trial"]["representative_history"]:
            end = labels.index(row["stage"])+1
            row["before"] = responses(problem, original[:, :end], original_force[:, :end], eta)
            row["after"] = responses(problem, field[:, :end], force[:, :end], eta)
    report["strip_edges_m"] = EDGES.tolist()
    report["strip_axes"] = ["direction", "lower_upper_depth", "strip"]
    report["checks"].update(saved_field_response_reconstruction=True, strip_endpoint_integral=True,
                            exact_control=True, constituent_force_balance=True)


def full_history_trial(problem, network, *, tolerance=.02):
    """Release every later field together through the existing causal map."""
    p = problem.physics
    eta = p.tensor([1., -1.])
    history = MaterialControlledHistory(problem, network, eta)
    products = dict(forward=0, transpose=0)

    def multiply(vector):
        products["forward"] += 1
        result = history.matvec(vector).ravel()
        print(json.dumps(dict(full_history_products=products)), flush=True)
        return result

    def transpose(vector):
        products["transpose"] += 1
        return history.rmatvec(vector).ravel()

    linear = LinearOperator((history.value.numel(), history.point.numel()),
        matvec=multiply, rmatvec=transpose, dtype=np.dtype(float))
    before = history.value.detach().numpy().ravel()
    solved = lsmr(linear, -before, atol=tolerance, btol=tolerance)
    direction, stop, iterations, residual_norm, normal_norm, operator_norm, condition, norm = solved
    image = linear@direction
    derivative = 2*before@image
    if derivative >= 0:
        raise RuntimeError("The complete history trial is not a descent direction")
    increment = p.tensor(direction.reshape(history.point.shape))
    with torch.no_grad():
        original, original_force = network(eta[None])
        factor = 1.
        initial_objective = float(before@before)
        while factor > np.finfo(float).eps:
            point = history.point+factor*increment
            after = history.function(point).numpy().ravel()
            if after@after <= initial_objective+1e-4*factor*derivative:
                break
            factor *= .5
        else:
            raise RuntimeError("The complete history trial line search failed")
        displacement = history.field(point)
        force = network.forces(displacement)
        np.testing.assert_array_equal(displacement[:, :1].numpy(), original[:, :1].numpy())
        h = 1e-4
        finite = ((history.function(history.point+h*increment)
            -history.function(history.point-h*increment))/(2*h)).numpy().ravel()
        check_directional_derivative(finite, image)
        labels = list(problem.boundary["stage_labels"])
        rows = []
        for name in ("rising_300kn", "global_peak", "event_1_after", "event_2_after", "falling_300kn"):
            s = labels.index(name)+1
            rows.append(dict(stage=str(name),
                before=responses(problem, original[:, :s], original_force[:, :s], eta[None]),
                after=responses(problem, displacement[:, :s], force[:, :s], eta[None])))
    report = dict(role="One nominal full-history neural-field diagnostic including the peak and both jumps; no per-material solution is used as a training label and no shared weights changed",
        later_spatial_coefficients=history.point.numel(), physical_steps=len(problem.initial),
        linear_tolerance=tolerance, stop_code=int(stop), iterations=int(iterations),
        accepted_scale=factor, objective_before=initial_objective,
        linearized_objective_after=float(np.sum((before+image)**2)), objective_after=float(after@after),
        residual_norm=float(residual_norm), normal_residual_norm=float(normal_norm),
        condition_estimate=float(condition), products=products, representative_history=rows,
        checks=dict(reference_preserved=True, full_causal_tangent=True))
    arrays = dict(history_trial_displacement_m=displacement[0].numpy(),
                  history_trial_force_n=force[0].numpy(), history_trial_raw_direction=direction)
    return report, arrays


def analyze(problem, network, *, step=-1):
    p = problem.physics
    eta = p.tensor([[1., -1.]])
    labels = list(map(str, problem.boundary["stage_labels"]))
    step %= len(labels)
    geometric, names = polynomial_fields(p.mesh)
    fields = p.tensor(geometric*1e-6)
    with torch.no_grad():
        original, original_force = network(eta)
        # Retain the full field before the chosen acquisition. The same balance
        # layer and full bulk/interface history determine the trial state.
        baseline = original[:, :step+1].clone()

    def evaluate(amplitudes):
        raw = torch.cat([baseline[:, :-1],
            baseline[:, -1:]+(amplitudes@fields)[None, None]], dim=1)
        displacement = network.balance(raw, eta)
        force = network.reference_force+(network.machine[None, 1:step+1]
            -(displacement[:, 1:]-displacement[:, :1])@p.jack_operator)/network.compliance
        force = torch.cat([network.reference_force.expand(1, 1), force], dim=1)
        residual, _ = problem.equilibrium(displacement, force, eta)
        return (residual[:, -1]@fields.T)[0], displacement, force

    point = fields.new_zeros(len(fields), requires_grad=True)
    value, _, _ = evaluate(point)
    tangent = []
    for i in range(len(fields)):
        tangent.append(torch.autograd.grad(value[i], point, retain_graph=True)[0].numpy())
        if (i+1) % 24 == 0:
            print(json.dumps(dict(stage=labels[step], tangent_rows=i+1, total=len(fields))), flush=True)
    tangent = np.asarray(tangent)
    before = value.detach().numpy()
    # A congruent diagonal scaling conditions only coordinates, not physics.
    scale = np.sqrt(np.maximum(np.abs(np.diag(tangent)), np.finfo(float).tiny))
    scaled = tangent/scale[:, None]/scale[None, :]
    amplitude = np.linalg.solve(scaled, -before/scale)/scale
    with torch.no_grad():
        after, displacement, force = evaluate(p.tensor(amplitude))
        # Verify the actual derivative at a perturbation smaller than the
        # branch-changing Newton step, rather than treating that step as linear.
        direction = p.tensor(amplitude/np.linalg.norm(amplitude))
        h = .0001
        finite = (evaluate(h*direction)[0]-evaluate(-h*direction)[0])/(2*h)
        image = tangent@direction.numpy()
        np.testing.assert_allclose(finite.numpy(), image, rtol=2e-4, atol=2e-7)
        np.testing.assert_allclose(displacement[:, :-1], baseline[:, :-1], rtol=0, atol=2e-13)
        original_responses = responses(problem, baseline, original_force[:, :step+1], eta)
        trial_responses = responses(problem, displacement, force, eta)
        # Recover exactly in the existing shared network's spatial coordinates.
        encoded = problem.coordinates.encode_nominal((amplitude@fields.numpy())[None])
        decoded = problem.coordinates.decode(p.tensor(encoded)[None])*p.scale
        np.testing.assert_allclose(decoded[0, 0].numpy(), amplitude@fields.numpy(), rtol=1e-9, atol=2e-13)
    result = dict(stage=labels[step], nominal_parameters=[1., -1.], mode_names=names,
        amplitude_unit="micrometre in the original polynomial field",
        amplitude=amplitude.tolist(), before=original_responses, after=trial_responses,
        before_virtual_work_j=before.tolist(), after_virtual_work_j=after.numpy().tolist(),
        tangent_storage="tangent_j in the sibling NPZ",
        scaled_tangent_condition_number=float(np.linalg.cond(scaled)),
        minimum_scaled_symmetric_eigenvalue=float(np.linalg.eigvalsh((scaled+scaled.T)/2).min()),
        checks=dict(causal_earlier_fields_preserved=True, mixed_physical_tangent=True,
                    existing_neural_space_reconstruction=True),
        role="One physical Newton direction at the selected acquisition; earlier full history retained. No trained weights or physical parameters changed.")
    arrays = dict(original_displacement_m=baseline[0].numpy(),
        trial_displacement_m=displacement[0].numpy(), trial_force_n=force[0].numpy(),
        polynomial_amplitude_um=amplitude, polynomial_fields=geometric,
        tangent_j=tangent, stage_labels=np.asarray(labels[:step+1]))
    return result, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="controlled_fracture_flexure")
    parser.add_argument("--stage", default="falling_300kn")
    parser.add_argument("--output", default="results/ideation/B2_postpeak_deformation.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--full-field", action="store_true")
    action.add_argument("--full-history", action="store_true")
    action.add_argument("--rebuild-responses", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    record = OUTPUT/f"{args.name}.json"
    saved = json.loads(record.read_text())
    problem = ControlledFractureProblem(saved["configuration"])
    network = problem.load_network(exact_control=True, fixed_reference=saved["fixed_reference"])
    network.load_state_dict(torch.load(record.with_suffix(".pt"), weights_only=True)["network"])
    step = list(problem.boundary["stage_labels"]).index(args.stage)
    if args.full_history and step != len(problem.initial)-1:
        parser.error("The full-history trial requires the final-acquisition baseline")
    if args.full_field or args.full_history or args.rebuild_responses:
        output = ROOT/args.output
        report = json.loads(output.read_text())
        arrays = dict(np.load(output.with_suffix(".npz")))
        if report["stage"] != args.stage or report["checkpoint"] != record.relative_to(ROOT).as_posix():
            raise ValueError("The saved polynomial trial belongs to another checkpoint or stage")
        with torch.no_grad():
            original, force = network(problem.physics.tensor([[1., -1.]]))
            baseline = original[:, :step+1]
            np.testing.assert_allclose(baseline[0].numpy(), arrays["original_displacement_m"], rtol=1e-10, atol=1e-13)
            report["before"] = responses(problem, baseline, force[:, :step+1], problem.physics.tensor([[1., -1.]]))
            report["after"] = responses(problem, problem.physics.tensor(arrays["trial_displacement_m"])[None],
                problem.physics.tensor(arrays["trial_force_n"])[None], problem.physics.tensor([[1., -1.]]))
        if args.rebuild_responses:
            rebuild_responses(problem, report, arrays)
            full_arrays = {}
        elif args.full_history:
            report["full_history_trial"], full_arrays = full_history_trial(problem, network)
        else:
            report["full_field_trial"], full_arrays = full_field_trial(problem, network, baseline)
        arrays.update(full_arrays)
        if not args.rebuild_responses:
            report["full_history_seconds" if args.full_history else "full_field_seconds"] = time.perf_counter()-started
    else:
        report, arrays = analyze(problem, network, step=step)
        report["elapsed_seconds"] = time.perf_counter()-started
    report["checkpoint"] = str(record.relative_to(ROOT)).replace("\\", "/")
    output = ROOT/args.output
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    states = dict(baseline=report["before"], polynomial=report["after"])
    if "full_field_trial" in report:
        states["full_field"] = report["full_field_trial"]["after"]
    if "full_history_trial" in report:
        states["full_history"] = report["full_history_trial"]["representative_history"][-1]["after"]
    keys = ("predicted_force_kn", "local_mean_increment_mm", "rod_total_kn")
    print(json.dumps(dict(saved=str(output), stage=report["stage"],
        responses={name: {key: state[key] for key in keys} for name, state in states.items()},
        seconds=time.perf_counter()-started)), flush=True)


if __name__ == "__main__":
    main()
