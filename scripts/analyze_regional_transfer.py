"""Resolve the saved path's load transfer and work on perimeter motions."""

import argparse
import json

import numpy as np
import torch

from controlled_fracture_history import ControlledFractureProblem, ROOT
from fixed_reference_history import OUTPUT
from regional_opening_basis import opening_fields
from spatial_fracture import elastic_stress, stress_components


STAGES = ("rising_300kn", "global_peak", "event_1_after", "event_2_after", "falling_300kn")
BANDS = (("core", 0., .33), ("perimeter", .33, .41),
         ("outer_observation", .41, .5), ("far_field", .5, np.inf))


def body_resultants(physics, force):
    nodal = force.reshape(*force.shape[:-1], -1, 3)
    return torch.stack([nodal[..., np.unique(physics.mesh["tetra"][physics.mesh["body"] == b]), :]
                        .sum(-2) for b in range(3)], dim=-2)


def resolve_terms(problem, displacement, force, parameters):
    """Signed residual terms; negate surface terms for force on a body."""
    p = problem.physics
    material = problem.base.materials(parameters)
    _, damage, history, parts = p.forces(displacement,
        compression_stiffness=material[:, 1, None, None], return_components=True)
    top_elastic = torch.sparse.mm(problem.base.top_stiffness,
        displacement.reshape(-1, p.ndof).T).T.reshape(displacement.shape)
    top, bulk_state = problem.fracture.force(displacement, material[:, 0])
    guide = p.tensor(p.mesh["lateral_guide_operator"])
    guides = (displacement@guide.T)@guide*p.settings["geometry"]["lateral_guide_stiffness_n_per_m"]
    terms = dict(top_bulk=top, substrate_cone_bulk=parts["bulk_and_guides"]-top_elastic-guides,
        interface=parts["interface"], cone_contact=parts["cone_contact"],
        trestle=parts["trestle"], anchors=p.anchor_loads(displacement)@p.anchor_operator,
        guides=guides, gravity=-p.gravity.expand_as(displacement),
        actuator=-force[..., None]*p.jack_operator)
    state = dict(interface_damage=damage, interface_history=history, **bulk_state)
    if problem.substrate_fracture is not None:
        correction, substrate = problem.substrate_correction(displacement)
        terms["substrate_cone_bulk"] = terms["substrate_cone_bulk"]+correction
        state.update({"substrate_"+name: value for name, value in substrate.items()})
    return terms, parts, state


def virtual_work(terms, fields):
    """Instantaneous signed force dot virtual displacement, in joules."""
    return {name: value@fields.T for name, value in terms.items()}


def surface_quadrature(physics, connectivity, area):
    points = np.einsum("qa,fac->fqc", physics.shape.numpy(), physics.mesh["xyz"][connectivity])
    weights = area[:, None]*physics.surface_weights.numpy()
    return points.reshape(-1, 3), weights.ravel()


def band_masks(points):
    radius = np.linalg.norm(points[:, :2], axis=-1)
    return [(name, (radius >= lower) & (radius < upper)) for name, lower, upper in BANDS]


def terminal_modal_trial(problem, network, displacement, eta, fields):
    """One local Newton direction in the saved field, with no weight update."""
    p = problem.physics

    def evaluate(amplitudes):
        field = torch.cat([displacement[:, :-1],
            displacement[:, -1:]+(amplitudes@fields)[None, None]], dim=1)
        field = network.balance(field, eta)
        force = network.forces(field)
        residual, state = problem.equilibrium(field, force, eta)
        return (residual[:, -1]@fields.T)[0], field, force, state

    zero = fields.new_zeros(len(fields), requires_grad=True)
    value = evaluate(zero)[0]
    tangent = torch.stack([torch.autograd.grad(value[i], zero, retain_graph=True)[0]
                           for i in range(len(fields))]).numpy()
    before = value.detach().numpy()
    amplitude = np.linalg.solve(tangent, -before)
    with torch.no_grad():
        after, trial, force, state = evaluate(p.tensor(amplitude))
        np.testing.assert_array_equal(trial[:, :-1].numpy(), displacement[:, :-1].numpy())
        host = p.observations(trial-displacement)[0][0, -1].numpy()*1e6
        local = (trial-displacement)@p.relative_operator.T
        original_damage = problem.fracture.stress_path(displacement, problem.base.materials(eta)[:, 0])[1]["damage"]
        # Check the tangent in a mixed direction to distinguish a nonlinear
        # trial outcome from an incorrectly differentiated balance layer.
        direction = p.tensor(np.arange(1., len(fields)+1.))
        direction /= torch.linalg.norm(direction)
        h = .001
        finite = (evaluate(h*direction)[0]-evaluate(-h*direction)[0])/(2*h)
        image = tangent@direction.numpy()
        if np.linalg.norm(finite.numpy()-image)/np.linalg.norm(image) >= 2e-4:
            raise AssertionError("The mixed modal tangent does not match the balanced finite difference")
    return dict(stage="falling_300kn", amplitude_unit="micrometre in each original geometric mode",
        role="Local direction diagnosis at the last state; earlier fields and histories retained, constituent motion and actuator force recomputed. No network weights or saved checkpoint changed.",
        tangent_j_per_unit=tangent.tolist(), tangent_condition_number=float(np.linalg.cond(tangent)),
        minimum_symmetric_tangent_eigenvalue=float(np.linalg.eigvalsh((tangent+tangent.T)/2).min()),
        amplitude=amplitude.tolist(), before_virtual_work_j=before.tolist(),
        linearized_virtual_work_j=(before+tangent@amplitude).tolist(), after_virtual_work_j=after.tolist(),
        predicted_force_kn=float(force[0, -1]/1e3),
        local_mean_increment_change_mm=float(local[0, -1].mean()*1e3),
        host_change_microstrain=host.tolist(), maximum_bulk_damage=float(state["bulk_damage"][0, -1].max()),
        maximum_bulk_damage_change=float((state["bulk_damage"]-original_damage).abs().max()),
        tangent_direction_check=True, earlier_fields_preserved=True)


def analyze(problem, network, saved_queries):
    p = problem.physics
    eta = p.tensor([[1., -1.]])
    labels = list(problem.boundary["stage_labels"])
    fields, names = opening_fields(p.mesh, p.settings["geometry"]["cone_top_radius_m"],
        problem.config["nonlocal_radius_m"], p.settings["observations"]["depths_m"],
        depth_kinematics="flexure")
    fields = p.tensor(fields*1e-6)
    with torch.no_grad():
        displacement, force = network(eta)
        residual, _ = problem.equilibrium(displacement, force, eta)
        terms, parts, state = resolve_terms(problem, displacement, force, eta)
        summed = sum(terms.values())
        torch.testing.assert_close(summed, residual, rtol=2e-10, atol=2e-7)
        nominal = int(np.flatnonzero(np.all(saved_queries["parameters"] == [1., -1.], axis=1))[0])
        outputs = p.mechanical_outputs(displacement)
        for key, actual in (("jack_force_n", force), ("rod_force_n", outputs["rod_force_n"]),
                            ("local_relative_m", outputs["local_relative_m"]),
                            ("maximum_bulk_damage", state["damage"].amax(-1))):
            np.testing.assert_allclose(actual[0].numpy(), saved_queries[key][nominal],
                                       rtol=1e-9, atol=1e-13)
        resultants = {key: body_resultants(p, value)[0].numpy() for key, value in terms.items()}
        work = {key: value[0].numpy() for key, value in virtual_work(terms, fields).items()}
        np.testing.assert_allclose(sum(work.values()), (residual@fields.T)[0].numpy(),
                                   rtol=2e-10, atol=1e-12)
        stress, _ = problem.fracture.stress_path(displacement, problem.base.materials(eta)[:, 0])
        trial_strain = problem.fracture.strains(fields[None])[0]
        volume = p.tensor(problem.fracture.volumes)
        density = torch.einsum("sqc,mqc,q->smq", stress[0], trial_strain, volume)
        np.testing.assert_allclose(density.sum(-1).numpy(), work["top_bulk"], rtol=2e-10, atol=1e-12)
        strain = problem.fracture.strains(displacement)
        effective = elastic_stress(strain, problem.base.materials(eta)[:, 0, None, None],
                                   p.settings["bulk"]["poisson"])
        eigenvalues, eigenvectors = torch.linalg.eigh(effective)
        negative = (eigenvectors*eigenvalues.clamp_max(0)[..., None, :])@eigenvectors.transpose(-1, -2)
        compressive = stress_components(negative)[0]
        tension = stress[0]-compressive
        bulk_spectral_work = {
            name: torch.einsum("sqc,mqc,q->smq", value, trial_strain, volume).numpy()
            for name, value in (("tensile", tension), ("compressive", compressive))}
        # A rigid correction has no bulk work. Constituent equilibrium also
        # makes its total virtual work zero at the later saved states.
        rigid_work = residual@network.constituent_balance.modes
        local = ((outputs["local_relative_m"]-outputs["local_relative_m"][:, :1])*1e3)[0].numpy()
        jack_increment = ((displacement-displacement[:, :1])@p.jack_operator)[0].numpy()*1e3

    interface_points, area = surface_quadrature(p, p.mesh["interface_upper"], p.mesh["interface_area"])
    cone_nodes = np.unique(p.mesh["tetra"][p.mesh["body"] == 2])
    lower_is_cone = np.repeat(np.isin(p.mesh["interface_lower"][:, 0], cone_nodes), len(p.surface_weights))
    traction = parts["interface_traction"][0].numpy().reshape(len(labels), -1, 3)
    jump = parts["interface_jump"][0].numpy().reshape(len(labels), -1, 3)
    damage = state["interface_damage"][0].numpy()
    bulk_damage = state["damage"][0].numpy()
    bulk_masks = band_masks(problem.fracture.points)
    interface_masks = band_masks(interface_points)
    density = density.numpy()
    nodal_fields = fields.numpy().reshape(len(fields), -1, 3)
    virtual_jump = np.einsum("qa,mfac->mfqc", p.shape.numpy(),
        nodal_fields[:, p.mesh["interface_upper"]]-nodal_fields[:, p.mesh["interface_lower"]])
    virtual_jump = virtual_jump.reshape(len(fields), -1, 3)
    surface_work = np.einsum("sqc,mqc,q->smqc", traction, virtual_jump, area)
    np.testing.assert_allclose(surface_work.sum(axis=(-1, -2)), work["interface"], rtol=2e-10, atol=1e-12)
    rows = []
    for name in STAGES:
        step = labels.index(name)
        interfaces = []
        for band, mask in interface_masks:
            for lower, lower_mask in (("cone", lower_is_cone), ("substrate", ~lower_is_cone)):
                selected = mask & lower_mask
                if not selected.any():
                    continue
                weight = area[selected]
                tr = traction[step, selected]
                closed = jump[step, selected, 2] < 0
                interfaces.append(dict(region=band, lower_body=lower, area_m2=float(weight.sum()),
                    physical_force_on_top_kn=(-np.sum(tr*weight[:, None], axis=0)/1e3).tolist(),
                    compression_uplift_kn=float(-np.minimum(tr[:, 2], 0)@weight/1e3),
                    tensile_downforce_kn=float(np.maximum(tr[:, 2], 0)@weight/1e3),
                    closed_area_fraction=float(weight@closed/weight.sum()),
                    damaged_area_fraction=float(weight@(damage[step, selected] > 0)/weight.sum()),
                    mean_damage=float(weight@damage[step, selected]/weight.sum()),
                    normal_virtual_work_j=surface_work[step][:, selected, 2].sum(-1).tolist(),
                    tangential_virtual_work_j=surface_work[step][:, selected, :2].sum(axis=(-1, -2)).tolist(),
                    maximum_opening_mm=float(np.maximum(jump[step, selected, 2], 0).max()*1e3),
                    maximum_slip_mm=float(np.linalg.norm(jump[step, selected, :2], axis=-1).max()*1e3)))
        bulk = []
        for band, mask in bulk_masks:
            weights = problem.fracture.volumes[mask]
            bulk.append(dict(region=band, volume_m3=float(weights.sum()),
                damaged_volume_fraction=float(weights@(bulk_damage[step, mask] > 0)/weights.sum()),
                mean_damage=float(weights@bulk_damage[step, mask]/weights.sum()),
                maximum_damage=float(bulk_damage[step, mask].max()),
                effective_maximum_tensile_stress_mpa=float(eigenvalues[0, step, mask, -1].max()/1e6),
                virtual_work_j=density[step][:, mask].sum(-1).tolist(),
                spectral_virtual_work_j={key: value[step][:, mask].sum(-1).tolist()
                                          for key, value in bulk_spectral_work.items()}))
        external_compliance = (force[0, step].item()-problem.reference_force)*problem.config["apparatus"]["compliance_m_per_n"]*1e3
        row = dict(stage=name, measured_force_kn=float(problem.boundary["force_n"][step]/1e3),
            predicted_force_kn=float(force[0, step]/1e3),
            local_mean_increment_mm=float(local[step].mean()), jack_increment_mm=float(jack_increment[step]),
            series_compliance_increment_mm=float(external_compliance),
            machine_increment_mm=float(problem.machine[step]*1e3),
            measured_local_mean_increment_mm=float(problem.boundary["measured_cone_relative_mean_increment_mm"][step]),
            measured_rod_total_increment_kn=float(problem.boundary["measured_rod_force_increment_kn"][step].sum()),
            predicted_rod_total_increment_kn=float((outputs["rod_force_n"][0, step]-outputs["rod_force_n"][0, 0]).sum()/1e3),
            rod_force_kn=(outputs["rod_force_n"][0, step].numpy()/1e3).tolist(),
            trestle_force_kn=float(outputs["trestle_force_n"][0, step]/1e3),
            body_residual_terms_kn={key: (value[step]/1e3).tolist() for key, value in resultants.items()},
            virtual_work_j={key: value[step].tolist() for key, value in work.items()},
            total_virtual_work_j=sum(work.values())[step].tolist(),
            interface_regions=interfaces, top_bulk_regions=bulk)
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("stage", "predicted_force_kn", "local_mean_increment_mm",
            "rod_force_kn", "trestle_force_kn", "total_virtual_work_j")}), flush=True)
    trial = terminal_modal_trial(problem, network, displacement, eta, fields)
    print(json.dumps(dict(terminal_modal_trial=trial)), flush=True)
    return dict(nominal_parameters=[1., -1.], physical_steps=len(labels),
        mode_names=names, virtual_amplitude_m=1e-6,
        body_order=["top", "substrate", "cone"],
        signs="Residual terms sum to R; physical surface forces on a body are the negatives of its residual terms. Positive z is upward.",
        work_definition="Instantaneous signed virtual work R dot v, without a factor 1/2; not stored energy, accumulated dissipation, or a solved finite motion.",
        radial_regions_m={name: [low, high if np.isfinite(high) else None] for name, low, high in BANDS},
        selection="Existing nominal material; rising 300 kN, measured peak, immediately after each observed jump, and falling 300 kN acquisition. Complete history is replayed before selecting these stages.",
        checks=dict(saved_nominal_reconstruction=True, force_term_sum=True, bulk_stress_virtual_work=True,
            virtual_work_sum=True, interface_quadrature_virtual_work=True,
            maximum_force_term_difference_n=float((summed-residual).abs().max()),
            maximum_later_scaled_rigid_virtual_work=float(rigid_work[:, 1:].abs().max())),
        representative_history=rows, terminal_modal_trial=trial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="controlled_fracture_flexure")
    parser.add_argument("--output", default="results/ideation/B2_regional_transfer.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    record = OUTPUT/f"{args.name}.json"
    saved = json.loads(record.read_text())
    problem = ControlledFractureProblem(saved["configuration"])
    network = problem.load_network(exact_control=True, fixed_reference=saved["fixed_reference"])
    network.load_state_dict(torch.load(record.with_suffix(".pt"), weights_only=True)["network"])
    queries = dict(np.load(record.with_name(record.stem+"_query.npz")))
    report = analyze(problem, network, queries)
    report["checkpoint"] = str(record.with_suffix(".pt").relative_to(ROOT)).replace("\\", "/")
    (ROOT/args.output).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
