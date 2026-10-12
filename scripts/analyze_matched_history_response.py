"""Read regional history contrasts from existing local-control equilibria.

The fixed-prefix queries are independent endpoints, while the accumulated
branch retains every intervening state. Both use the same material, initial
reference and installed circuit. No equilibrium solve or circuit selection is
performed. Prefix swaps below are constitutive diagnostics at a held field;
they do not construct additional equilibria or experimental loading paths.
"""

import json

import numpy as np
import torch

from analyze_history_design_value import layers, rms
from controlled_fracture_history import ControlledFractureProblem, ROOT
from fixed_reference_history import OUTPUT
from weighted_readout import components, shunt_design, uniform_parameters, weighted_response


FIXED = ROOT/"results/ideation/B2_controlled_endpoint_search.json"
ACCUMULATED = ROOT/"results/ideation/B2_controlled_reachability.json"
DESIGN = OUTPUT/"regional_history_readout.json"
RESULT = ROOT/"results/ideation/B2_matched_history_response.json"


def matched_indices(targets_mm, coordinates_mm):
    """Match saved controls without interpolating or merging path branches."""
    pairs = []
    for fixed_index, target in enumerate(targets_mm):
        matches = np.flatnonzero(np.isclose(coordinates_mm, target, rtol=0, atol=1e-9))
        if len(matches) > 1:
            raise ValueError(f"The accumulated path has multiple states at {target} mm")
        if len(matches):
            pairs.append((fixed_index, int(matches[0])))
    return pairs


def reference_readout(raw, parameters, settings):
    """Apply the finite circuit before subtracting its first, loaded reference."""
    circuit = weighted_response(raw, parameters, settings)
    return circuit["readout"]-circuit["readout"][:1], circuit


def sign_differences(host, circuit):
    return [dict(index=index.tolist(), host_microstrain=float(host[tuple(index)]),
                 circuit_microstrain=float(circuit[tuple(index)]))
            for index in np.argwhere(np.sign(host) != np.sign(circuit))]


def regional_contrast(host, readout):
    """Keep mean, depth, layer and left/right differences separately legible."""
    h, c = host[1]-host[0], readout[1]-readout[0]
    return dict(host_microstrain=h.tolist(), circuit_microstrain=c.tolist(),
        host_layer_microstrain=layers(h).tolist(), circuit_layer_microstrain=layers(c).tolist(),
        host_right_minus_left_microstrain=(h[:, 2]-h[:, 0]).tolist(),
        circuit_right_minus_left_microstrain=(c[:, 2]-c[:, 0]).tolist(),
        host_rms_microstrain=rms(h), circuit_rms_microstrain=rms(c),
        synthesis_difference_rms_microstrain=rms(c-h),
        component_sign_differences=sign_differences(h, c),
        layer_sign_differences=sign_differences(layers(h), layers(c)))


@torch.no_grad()
def material_history(problem, fields, eta):
    """Replay the existing constitutive laws, excluding apparatus residuals."""
    p = problem.physics
    u = p.tensor(fields)[None]
    material = problem.base.materials(p.tensor(eta)[None])
    stress, bulk = problem.fracture.stress_path(u, material[:, 0])
    nodal = u.reshape(1, len(fields), -1, 3)
    jump = torch.einsum("qa,bsfac->bsfqc", p.shape,
                       nodal[:, :, p.upper]-nodal[:, :, p.lower]).flatten(2, 3)
    traction, damage, history = p.interface_path(jump, material[:, 1, None, None])
    # The selected source has the original, frictionless interface law.
    if p.interface_friction_coefficient:
        raise ValueError("This matched-history analysis expects the original frictionless scenario")
    return dict(bulk_stress=stress[0].numpy(),
        bulk_damage=bulk["damage"][0].numpy(), bulk_history=bulk["history"][0].numpy(),
        bulk_current=bulk["nonlocal_strain"][0].numpy(),
        interface_traction=traction[0].numpy(), interface_damage=damage[0].numpy(),
        interface_history=history[0].numpy())


def terminal(history, index=-1):
    return {key: value[index] for key, value in history.items()}


def damage_summary(state, volume, area):
    return dict(bulk_mean_damage=float(state["bulk_damage"]@volume/volume.sum()),
        interface_mean_damage=float(state["interface_damage"]@area/area.sum()),
        bulk_prior_maximum_active_volume_fraction=float(
            ((state["bulk_history"] > state["bulk_current"])
             & (state["bulk_damage"] > 0))@volume/volume.sum()))


def prefix_effect(short, long, volume, area):
    """Added intermediate states at exactly the same final displacement field."""
    bulk = long["bulk_damage"]-short["bulk_damage"]
    interface = long["interface_damage"]-short["interface_damage"]
    if min(bulk.min(), interface.min()) < -1e-12:
        raise AssertionError("Adding the accumulated prefix reduced irreversible damage")
    stress = long["bulk_stress"]-short["bulk_stress"]
    traction = long["interface_traction"]-short["interface_traction"]
    return dict(added_bulk_mean_damage=float(bulk@volume/volume.sum()),
        added_interface_mean_damage=float(interface@area/area.sum()),
        bulk_stress_change_volume_rms_mpa=float(np.sqrt(
            ((stress**2)@np.array([1., 1., 1., 2., 2., 2.]))@volume/volume.sum())/1e6),
        interface_traction_change_area_rms_mpa=float(np.sqrt(
            (traction**2).sum(-1)@area/area.sum())/1e6))


@torch.no_grad()
def analyze():
    torch.set_num_threads(2)
    fixed_report = json.loads(FIXED.read_text())
    accumulated_report = json.loads(ACCUMULATED.read_text())
    parent_report = json.loads((ROOT/accumulated_report["source_path"]).read_text())
    accumulated_stages = {row["index"]: row for report in (parent_report, accumulated_report)
                          for row in report["stages"]}
    fixed = dict(np.load(FIXED.with_suffix(".npz")))
    accumulated = dict(np.load(ACCUMULATED.with_suffix(".npz")))
    checkpoint = json.loads((ROOT/fixed_report["checkpoint"]).read_text())
    configuration = fixed_report.get("configuration", checkpoint["configuration"])
    if configuration != accumulated_report.get("configuration", checkpoint["configuration"]):
        raise ValueError("Matched histories must use the same physical configuration")
    np.testing.assert_array_equal(fixed_report["nominal_parameters"], accumulated_report["nominal_parameters"])
    problem = ControlledFractureProblem(configuration)
    p = problem.physics
    eta = fixed_report["nominal_parameters"]
    count = fixed_report["prefix_index"]+1
    prefix = fixed["displacement_m"][:count]
    np.testing.assert_array_equal(prefix, accumulated["displacement_m"][:count])
    np.testing.assert_array_equal(fixed["force_n"][:count], accumulated["force_n"][:count])
    np.testing.assert_array_equal(fixed["machine_increment_m"][:count], accumulated["machine_increment_m"][:count])
    reference = prefix[0]
    local_operator = p.relative_operator.mean(dim=0).numpy()
    coordinates = (accumulated["displacement_m"]-reference)@local_operator*1e3
    targets = [row["target_local_increment_mm"] for row in fixed_report["stages"]]
    pairs = matched_indices(targets, coordinates)
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    design = json.loads(DESIGN.read_text())["designs"]["selected"]["parameters"]
    fields = np.stack([reference]+[field for i, j in pairs for field in
        (fixed["numerical_endpoint_m"][i], accumulated["displacement_m"][j])])
    host = p.observations(p.tensor(fields-reference))[0].numpy()*1e6
    raw = fields@problem.base.raw_matrix.numpy().T
    readout, circuit = reference_readout(raw, design, settings)
    readout *= 1e6
    # Trace any changed regional contrast through the spatial sampling and
    # installed weights before assigning it to finite circuit nonlinearity.
    uniform_weights = shunt_design(uniform_parameters(), settings)[0]
    sampled = components(raw@uniform_weights.T)*1e6
    installed_linear = components(raw@circuit["weights"].T)*1e6
    accumulated_raw = accumulated["displacement_m"]@problem.base.raw_matrix.numpy().T
    prefix_gauge_max = float(np.abs(accumulated_raw[:count]).max()*1e6)
    compliance = configuration["apparatus"]["compliance_m_per_n"]
    mechanism_targets = {0.8, 2., 3.}
    history = material_history(problem, accumulated["displacement_m"], eta)
    volume = problem.fracture.volumes
    area = (p.area[:, None]*p.surface_weights).numpy().ravel()
    result = []
    for pair_index, (i, j) in enumerate(pairs):
        target = targets[i]
        rows = np.array([1+2*pair_index, 2+2*pair_index])
        controls = [fixed["numerical_machine_increment_m"][i], accumulated["machine_increment_m"][j]]
        response = []
        for state_index, (row, machine) in enumerate(zip(rows, controls, strict=True)):
            field = fields[row]
            local = (field-reference)@local_operator*1e3
            np.testing.assert_allclose(local, target, rtol=0, atol=1e-9)
            force = problem.reference_force+(machine-(field-reference)@p.jack_operator.numpy())/compliance
            mechanical = p.mechanical_outputs(p.tensor(field))
            response.append(dict(local_increment_mm=float(local), machine_increment_mm=float(machine*1e3),
                force_kn=float(force/1e3), rod_total_kn=float(mechanical["rod_force_n"].sum()/1e3),
                trestle_total_kn=float(mechanical["trestle_force_n"]/1e3),
                host_microstrain=host[row].tolist(), circuit_microstrain=readout[row].tolist(),
                gauge_abs_max_microstrain=float(np.abs(raw[row]).max()*1e6),
                history_gauge_abs_max_microstrain=(max(prefix_gauge_max, float(np.abs(raw[row]).max()*1e6))
                    if state_index == 0 else float(np.abs(accumulated_raw[:j+1]).max()*1e6)),
                gauge_indices_above_nominal_limit=np.flatnonzero(
                    np.abs(raw[row])*1e6 > settings["sensor_nominal_strain_limit_microstrain"]).tolist(),
                amplifier_voltage_range_v=[float(circuit["voltage"][row].min()),
                                           float(circuit["voltage"][row].max())]))
        expected = fixed_report["stages"][i]["after"]
        np.testing.assert_allclose(response[0]["force_kn"], expected["predicted_force_kn"], rtol=1e-9)
        np.testing.assert_allclose(host[rows[0]], expected["host_microstrain"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(response[1]["force_kn"], accumulated["force_n"][j]/1e3, rtol=1e-9)
        np.testing.assert_allclose(host[rows[1]], accumulated_stages[j]["after"]["host_microstrain"],
                                   rtol=1e-9, atol=1e-9)
        case = dict(local_increment_mm=target, fixed_numerical_index=i, accumulated_index=j,
            history_state_counts=[count+1, j+1], fixed_prefix=response[0], accumulated=response[1],
            accumulated_minus_fixed=regional_contrast(host[rows], readout[rows]))
        case["observation_chain_contrast_microstrain"] = dict(
            uniform_finite_gauges=(sampled[rows[1]]-sampled[rows[0]]).tolist(),
            installed_linear_weights=(installed_linear[rows[1]]-installed_linear[rows[0]]).tolist(),
            finite_circuit_minus_installed_linear=(readout[rows[1]]-readout[rows[0]]
                -installed_linear[rows[1]]+installed_linear[rows[0]]).tolist())
        if target in mechanism_targets:
            # Each fixed query consists only of the common prefix and this endpoint.
            fixed_history = terminal(material_history(problem, np.vstack([prefix, fields[rows[0]]]), eta))
            accumulated_history = terminal(history, j)
            np.testing.assert_allclose(fixed_history["bulk_damage"].max(), expected["maximum_bulk_damage"],
                                       rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(accumulated_history["bulk_damage"].max(),
                accumulated_stages[j]["after"]["maximum_bulk_damage"], rtol=1e-9, atol=1e-12)
            fixed_with_long_prefix = terminal(material_history(problem,
                np.vstack([accumulated["displacement_m"][:j], fields[rows[0]]]), eta))
            accumulated_with_short_prefix = terminal(material_history(problem,
                np.vstack([prefix, fields[rows[1]]]), eta))
            case["material_history"] = dict(
                fixed_prefix=damage_summary(fixed_history, volume, area),
                accumulated=damage_summary(accumulated_history, volume, area),
                adding_intermediate_history_at_fixed_endpoint=prefix_effect(
                    fixed_history, fixed_with_long_prefix, volume, area),
                adding_intermediate_history_at_accumulated_endpoint=prefix_effect(
                    accumulated_with_short_prefix, accumulated_history, volume, area))
        result.append(case)
        print(json.dumps(dict(local_mm=target,
            host_contrast_rms=case["accumulated_minus_fixed"]["host_rms_microstrain"],
            circuit_contrast_rms=case["accumulated_minus_fixed"]["circuit_rms_microstrain"],
            gauge_max_microstrain=[r["gauge_abs_max_microstrain"] for r in response])), flush=True)
    report = dict(
        sources=dict(fixed_prefix=FIXED.relative_to(ROOT).as_posix(),
            accumulated=ACCUMULATED.relative_to(ROOT).as_posix(),
            circuit=DESIGN.relative_to(ROOT).as_posix(), checkpoint=fixed_report["checkpoint"]),
        role="Matched local-control states from two conditional numerical histories; not distinct measured histories or a high-load neural validation",
        contrast_order="accumulated minus fixed prefix",
        axes=dict(directions=["1", "2"], regions=["left", "central", "right"],
                  components=["mean", "half_top_minus_bottom"], layers=["bottom", "top"]),
        nominal_material_parameters=eta, common_prefix_states=count,
        common_prefix_final_local_mm=float(coordinates[count-1]),
        fixed_circuit_parameters=design,
        circuit_role="The low-load regional-history design is installed unchanged; its high-load response is an extrapolated algebraic circuit calculation",
        gauge_nominal_limit_microstrain=settings["sensor_nominal_strain_limit_microstrain"],
        prefix_swap_role="Hold the full endpoint displacement fixed and add intermediate history. Stress/traction changes isolate constitutive memory, but swapped fields are not re-equilibrated",
        cases=result)
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    analyze()
