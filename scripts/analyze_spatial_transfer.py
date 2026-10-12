"""Locate target-information loss along the existing physical observation chain.

The prescribed Fourier directions are fixed by inputs/spatial_transfer.json.
The full path Jacobian is evaluated at independent saved neural predictions.
No spatially varied primal trajectory or separate reference solver is used.
"""

import json

import numpy as np
import torch

from functional_readout import PathLinearization, ReadoutFunctionals
from stateful_physics import ROOT, MetricPathNetwork, line_average_operator, load_problem


OUTPUT = ROOT/"results/ideation/stateful_pilot/spatial_transfer"
REFERENCES = ROOT/"results/ideation/stateful_pilot/functional/validation"


def bond_basis(physics, settings):
    """Non-redundant real Fourier basis at the actual interface quadrature."""
    xy = np.einsum("qa,fac->fqc", physics.shape.cpu().numpy(),
                   physics.mesh["xyz"][physics.mesh["interface_upper"], :2]).reshape(-1, 2)
    columns, names, orders = [np.ones(len(xy))], ["bond_mean"], [0]
    order = settings["maximum_frequency_index"]
    for m in range(order+1):
        for n in range(-order, order+1):
            if m == 0 and n <= 0:
                continue
            phase = 2*np.pi*(m*xy[:, 0]+n*xy[:, 1])/settings["spatial_period_m"]
            for kind, value in [("cos", np.cos(phase)), ("sin", np.sin(phase))]:
                columns.append(value)
                names.append(f"bond_{kind}_{m}_{n}")
                orders.append(max(m, abs(n)))
    return np.stack(columns, axis=1), names, np.asarray(orders)


def raw_gauges(physics):
    """Keep each finite gauge before the current series-string averaging."""
    obs = physics.settings["observations"]
    circuit = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    matrices, groups = [], []
    for candidate in obs["candidates"]:
        rows, labels = [], []
        for direction in obs["directions"]:
            for r, (left, right) in enumerate(obs["regions_m"]):
                count = circuit["regions"][r]["gauges_per_depth"]
                centres = left+(right-left)*(np.arange(count)+0.5)/count
                for depth, z in enumerate(obs["depths_m"]):
                    for centre in centres+candidate["shift_m"]:
                        start, end = np.array([0., 0., z]), np.array([0., 0., z])
                        start[direction] = centre-candidate["length_m"]/2
                        end[direction] = centre+candidate["length_m"]/2
                        rows.append(line_average_operator(physics.mesh, physics.inverse,
                                                         start, end, direction))
                        labels.append((direction*3+r)*2+depth)
        matrices.append(np.stack(rows))
        groups.append(np.stack([np.asarray(labels) == k for k in range(12)]).astype(float))
        groups[-1] /= groups[-1].sum(axis=1, keepdims=True)
    return np.stack(matrices), np.stack(groups)


def prescribed_directions(physics, displacement, settings, basis):
    """R_eta at fixed free displacements, including strong driving conditions."""
    p = physics
    base = displacement[None]
    driven = torch.zeros_like(base)
    driven[:, :, p.driven] = (settings["amplitude_relative_scale"]*p.path[None, :, None]*p.scale)
    _, amplitude_force = torch.autograd.functional.jvp(lambda u: p.forces(u)[0], base, driven)
    force_columns = [amplitude_force[0, :, p.free].cpu().numpy()]
    scale = torch.ones((1, 1, basis.shape[0]), dtype=p.dtype, device=p.device)
    for column in basis.T:
        direction = p.tensor(settings["bond_relative_scale"]*column)[None, None]
        _, force = torch.autograd.functional.jvp(lambda s: p.forces(base, bond_scale=s)[0],
                                                scale, direction)
        force_columns.append(force[0, :, p.free].cpu().numpy())
    explicit = np.zeros((*displacement.shape, len(force_columns)))
    explicit[:, :, 0] = driven[0].cpu().numpy()
    return np.stack(force_columns, axis=-1), explicit


def kernel_check(observed, target):
    """Exact local algebra at floating precision, with no scientific cutoff.

    A null direction of observed cannot generate a target in any subsequent
    differentiable readout. Singular values remain explicit: numerical rank
    is not a noise-resolution claim.
    """
    _, singular, vh = np.linalg.svd(observed, full_matrices=True)
    tolerance = max(observed.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    rank = int(np.count_nonzero(singular > tolerance))
    null = vh[rank:].T
    invisible = target@null
    norm = np.linalg.norm(target)
    ratio = float(np.linalg.norm(invisible)/norm) if norm else 0.
    if null.shape[1] and np.linalg.norm(invisible) > 0:
        _, missed, right = np.linalg.svd(invisible, full_matrices=False)
        witness = null@right[0]
        # This witness stays in the field's declared finite l1 neighborhood.
        witness /= max(abs(witness[0]), np.sum(np.abs(witness[1:])), 1.)
        observed_witness, target_witness = observed@witness, target@witness
    else:
        missed = np.array([0.])
        witness = np.zeros(observed.shape[1])
        observed_witness, target_witness = observed@witness, target@witness
    row_map = target@np.linalg.pinv(observed, rcond=max(observed.shape)*np.finfo(float).eps)
    return dict(rank=rank, parameter_count=observed.shape[1], observation_count=observed.shape[0],
        nullity=observed.shape[1]-rank, singular_values_microstrain=singular.tolist(),
        target_outside_observed_rowspace_fraction=ratio,
        largest_invisible_target_singular_microstrain=float(missed[0]),
        witness_parameter_vector=witness.tolist(),
        witness_observed_rms_microstrain=float(np.sqrt(np.mean(observed_witness**2))),
        witness_target_rms_microstrain=float(np.sqrt(np.mean(target_witness**2))),
        witness_target_microstrain=target_witness.tolist(),
        unconstrained_linear_map_spectral_gain=float(np.linalg.norm(row_map, ord=2)))


def reflection_subspaces(names, order):
    """Separate sin(mx)sin(ny), which vanishes on both ideal symmetry axes.

    Geometry/constraints and the triangulation of this pilot are not exactly
    reflection symmetric. Removing these four parameter combinations tests
    whether the observed grouping loss relies on their spurious axis response.
    The transformation is orthogonal and is not selected from sensor outputs.
    """
    identity = np.eye(len(names))
    used, odd, even = set(), [], []
    for m in range(1, order+1):
        for n in range(1, order+1):
            positive = names.index(f"bond_cos_{m}_{n}")
            negative = names.index(f"bond_cos_{m}_{-n}")
            odd.append((identity[:, positive]-identity[:, negative])/np.sqrt(2))
            even.append((identity[:, positive]+identity[:, negative])/np.sqrt(2))
            used.update([positive, negative])
    retained = np.stack([identity[:, i] for i in range(len(names)) if i not in used]+even, axis=1)
    double_odd = np.stack(odd, axis=1)
    transform = np.concatenate([retained, double_odd], axis=1)
    np.testing.assert_allclose(transform.T@transform, identity, atol=1e-15)
    return retained, double_odd


def inspect_reference(physics, settings, basis, names, orders, raw, groups, name):
    record = json.loads((REFERENCES/f"{name}.json").read_text())
    network = MetricPathNetwork(physics, record["reference_settings"]["width"])
    saved = torch.load(REFERENCES/f"{name}.pt", map_location=physics.device, weights_only=True)
    network.load_state_dict(saved["network"])
    amplitude = physics.tensor([settings["nominal_amplitude"]])
    with torch.no_grad():
        u = physics.displacements(network(amplitude), amplitude)[0]
    linear = PathLinearization(physics, u)
    partial, tangent = prescribed_directions(physics, u, settings, basis)
    tangent[:, physics.free.cpu().numpy()] = linear.solve(-partial)
    functionals = ReadoutFunctionals(physics)
    q = np.einsum("od,sdp->sop", functionals.q_matrix.cpu(), tangent)*1e6
    e_raw = np.einsum("cod,sdp->csop", raw, tangent)*1e6
    e_string = np.einsum("crg,csgp->csrp", groups, e_raw)
    gradient = functionals.gradients(u)
    f = np.einsum("sod,sdp->sop", gradient[:, 12:],
                  tangent[:, physics.free.cpu().numpy()])*1e6
    # Observations have no support on the strongly driven cone bottom.
    f = f.reshape(len(u), len(raw), 12, -1).transpose(1, 0, 2, 3)
    checks = {}
    for label, mask in [("bay_scale", np.r_[True, orders <= 1]),
                        ("through_half_bay", np.ones(len(names), dtype=bool))]:
        target = q[:, :, mask]
        entries = []
        for ci, candidate in enumerate(physics.settings["observations"]["candidates"]):
            channels = {}
            for stage, values in [("individual_gauges", e_raw[ci]),
                                  ("series_strings", e_string[ci]), ("loaded_amplifier", f[ci])]:
                observed = values[:, :, mask]
                channels[stage] = dict(per_step=[kernel_check(e, h) for e, h in zip(observed, target)],
                    complete_known_path=kernel_check(observed.reshape(-1, mask.sum()),
                                                    target.reshape(-1, mask.sum())))
            entries.append(dict(candidate=candidate["name"], channels=channels))
        checks[label] = dict(parameter_names=np.asarray(names)[mask].tolist(), candidates=entries)
    report = dict(reference=name, nominal_amplitude=settings["nominal_amplitude"],
        nonzero_path=physics.path.cpu().tolist(), checks=checks)
    arrays = dict(displacement_m=u.cpu().numpy(), displacement_tangent_m=tangent,
                  parameter_force_n=partial, target_tangent_microstrain=q,
                  raw_gauge_tangent_microstrain=e_raw, string_tangent_microstrain=e_string,
                  readout_tangent_microstrain=f, parameter_names=np.asarray(names))
    return arrays, report


def main():
    torch.set_num_threads(4)
    settings = json.loads((ROOT/"inputs/spatial_transfer.json").read_text())
    physics = load_problem()
    physics.prepare_elastic_metric()
    basis, bond_names, orders = bond_basis(physics, settings)
    names = ["loading_amplitude", *bond_names]
    raw, groups = raw_gauges(physics)
    np.testing.assert_allclose(np.einsum("crg,cgd->crd", groups, raw),
                               physics.candidates.cpu(), rtol=1e-12, atol=1e-12)
    OUTPUT.mkdir(exist_ok=True)
    saved_arrays, reports = [], []
    for name in settings["references"]:
        arrays, report = inspect_reference(physics, settings, basis, names, orders, raw, groups, name)
        np.savez_compressed(OUTPUT/f"{name}.npz", **arrays)
        (OUTPUT/f"{name}.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        saved_arrays.append(arrays)
        reports.append(report)
        print(json.dumps(dict(reference=name, completed=True)), flush=True)
    agreement = {}
    for key in ["target_tangent_microstrain", "raw_gauge_tangent_microstrain",
                "string_tangent_microstrain", "readout_tangent_microstrain"]:
        a, b = [r[key] for r in saved_arrays]
        agreement[key] = dict(rms_difference=float(np.sqrt(np.mean((a-b)**2))),
                             rms_reference_a=float(np.sqrt(np.mean(a*a))))
    cross_witnesses = []
    candidates = reports[0]["checks"]["through_half_bay"]["candidates"]
    for ci, candidate in enumerate(candidates):
        per_step = []
        for step, check in enumerate(candidate["channels"]["loaded_amplifier"]["per_step"]):
            witness = np.asarray(check["witness_parameter_vector"])
            evaluations = []
            for name, arrays in zip(settings["references"], saved_arrays):
                observed = arrays["readout_tangent_microstrain"][ci, step]@witness
                target = arrays["target_tangent_microstrain"][step]@witness
                evaluations.append(dict(reference=name,
                    observed_rms_microstrain=float(np.sqrt(np.mean(observed**2))),
                    target_rms_microstrain=float(np.sqrt(np.mean(target**2)))))
            per_step.append(evaluations)
        cross_witnesses.append(dict(candidate=candidate["candidate"], per_step=per_step))
    retained, double_odd = reflection_subspaces(names, settings["maximum_frequency_index"])
    symmetry = []
    for name, arrays in zip(settings["references"], saved_arrays):
        target = arrays["target_tangent_microstrain"]
        q_retained = target@retained
        entries = []
        for ci, candidate in enumerate(candidates):
            observed = arrays["readout_tangent_microstrain"][ci]@retained
            checks = []
            for e, q in zip(observed, q_retained):
                check = kernel_check(e, q)
                checks.append({key: check[key] for key in ["rank", "parameter_count", "nullity",
                    "singular_values_microstrain", "target_outside_observed_rowspace_fraction"]})
            entries.append(dict(candidate=candidate["candidate"], per_step=checks))
        symmetry.append(dict(reference=name, retained_parameter_count=retained.shape[1],
            double_odd_target_rms_microstrain_by_step=np.sqrt(np.mean((target@double_odd)**2,
                                                                      axis=(1, 2))).tolist(),
            candidates=entries))
    summary = dict(settings=settings, parameter_names=names, reference_agreement=agreement,
        reference_a_null_directions_on_both_references=cross_witnesses,
        reflection_subspace_check=symmetry,
        interpretation="Local full-history implicit tangents at saved PINN predictions. Per-step kernel checks concern one inquiry; the complete-path check assumes this entire known history is available. Singular values and cross-reference differences delimit interpretation; no measurement-noise threshold is inferred.")
    (OUTPUT/"comparison.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(agreement), flush=True)


if __name__ == "__main__":
    main()
