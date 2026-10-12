"""Read one causal unloading/reloading path through the installed circuit.

Repeated local controls retain their chronological identities. Each new state
is compared with its first passage in the saved loading prefix; circuit output
is referenced to the original loaded state after evaluating the finite bridge.
"""

import json

import numpy as np
import torch

from analyze_controlled_capacity import controlled_forces
from analyze_matched_history_response import (DESIGN, damage_summary, material_history,
    prefix_effect, reference_readout, regional_contrast, terminal)
from controlled_fracture_history import ControlledFractureProblem, ROOT
from weighted_readout import components, shunt_design, uniform_parameters


SOURCE = ROOT/"results/ideation/B2_unloading_history.json"
RESULT = ROOT/"results/ideation/B2_unloading_history_response.json"


def branch_pairs(coordinates_mm, prefix_index, stages):
    """Match each continued passage only against the unchanged loading prefix."""
    coordinates = np.asarray(coordinates_mm)
    pairs = []
    for stage in stages:
        later = stage["index"]
        target = stage["target_local_increment_mm"]
        if later <= prefix_index:
            raise ValueError("A continued state must follow the source prefix")
        np.testing.assert_allclose(coordinates[later], target, rtol=0, atol=1e-9)
        matches = np.flatnonzero(np.isclose(coordinates[:prefix_index+1], target,
                                           rtol=0, atol=1e-9))
        if len(matches) > 1:
            raise ValueError(f"The loading prefix has multiple states at {target} mm")
        if len(matches):
            pairs.append((int(matches[0]), later))
    return pairs


@torch.no_grad()
def analyze():
    torch.set_num_threads(2)
    path_report = json.loads(SOURCE.read_text())
    if path_report["termination"] != "local_targets_completed":
        raise ValueError("Complete the requested unloading/reloading path before analysis")
    source_path = ROOT/path_report["source_path"]
    source_report = json.loads(source_path.read_text())
    parent_report = json.loads((ROOT/source_report["source_path"]).read_text())
    saved_stages = {row["index"]: row for report in (parent_report, source_report, path_report)
                    for row in report["stages"]}
    arrays = dict(np.load(SOURCE.with_suffix(".npz")))
    source = dict(np.load(source_path.with_suffix(".npz")))
    checkpoint = json.loads((ROOT/path_report["checkpoint"]).read_text())
    configuration = path_report.get("configuration", checkpoint["configuration"])
    if configuration != source_report.get("configuration", checkpoint["configuration"]):
        raise ValueError("The loading and continued paths must use the same configuration")
    np.testing.assert_array_equal(path_report["nominal_parameters"], source_report["nominal_parameters"])
    count = path_report["prefix_index"]+1
    for name in ("displacement_m", "machine_increment_m", "force_n"):
        np.testing.assert_array_equal(arrays[name][:count], source[name][:count])
    problem = ControlledFractureProblem(configuration)
    p = problem.physics
    eta = path_report["nominal_parameters"]
    fields = arrays["displacement_m"]
    reference = fields[0]
    problem.machine = p.tensor(arrays["machine_increment_m"])
    np.testing.assert_allclose(controlled_forces(problem, p.tensor(fields)).numpy(),
                               arrays["force_n"], rtol=1e-13, atol=1e-7)
    coordinates = (fields-reference)@p.relative_operator.mean(dim=0).numpy()*1e3
    pairs = branch_pairs(coordinates, count-1, path_report["stages"])
    settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
    design = json.loads(DESIGN.read_text())["designs"]["selected"]["parameters"]
    host = p.observations(p.tensor(fields-reference))[0].numpy()*1e6
    raw = fields@problem.base.raw_matrix.numpy().T
    readout, circuit = reference_readout(raw, design, settings)
    readout *= 1e6
    uniform_weights = shunt_design(uniform_parameters(), settings)[0]
    sampled = components(raw@uniform_weights.T)*1e6
    installed_linear = components(raw@circuit["weights"].T)*1e6
    nominal_limit = settings["sensor_nominal_strain_limit_microstrain"]
    history = material_history(problem, fields, eta)
    original_history = material_history(problem, source["displacement_m"][:count], eta)
    for key, value in original_history.items():
        np.testing.assert_allclose(history[key][:count], value, rtol=1e-12, atol=1e-12)
    for name in ("bulk_history", "interface_history"):
        if np.any(history[name][1:] < history[name][:-1]):
            raise AssertionError(f"The continued path lost {name}")
    del original_history
    volume = problem.fracture.volumes
    area = (p.area[:, None]*p.surface_weights).numpy().ravel()

    def response(index):
        mechanical = p.mechanical_outputs(p.tensor(fields[index]))
        result = dict(index=index, local_increment_mm=float(coordinates[index]),
            machine_increment_mm=float(arrays["machine_increment_m"][index]*1e3),
            force_kn=float(arrays["force_n"][index]/1e3),
            rod_total_kn=float(mechanical["rod_force_n"].sum()/1e3),
            trestle_total_kn=float(mechanical["trestle_force_n"]/1e3),
            host_microstrain=host[index].tolist(), circuit_microstrain=readout[index].tolist(),
            gauge_abs_max_microstrain=float(np.abs(raw[index]).max()*1e6),
            history_gauge_abs_max_microstrain=float(np.abs(raw[:index+1]).max()*1e6),
            gauge_indices_above_nominal_limit=np.flatnonzero(
                np.abs(raw[index])*1e6 > nominal_limit).tolist(),
            amplifier_voltage_range_v=[float(circuit["voltage"][index].min()),
                                       float(circuit["voltage"][index].max())])
        expected = saved_stages[index]["after"]
        np.testing.assert_allclose(result["force_kn"], expected["predicted_force_kn"], rtol=1e-9)
        np.testing.assert_allclose(host[index], expected["host_microstrain"], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(history["bulk_damage"][index].max(),
                                   expected["maximum_bulk_damage"], rtol=1e-9, atol=1e-12)
        return result

    cases = []
    for first, later in pairs:
        rows = np.array([first, later])
        first_state, later_state = terminal(history, first), terminal(history, later)
        first_with_later_prefix = terminal(material_history(problem,
            np.vstack([fields[:later], fields[first]]), eta))
        later_with_first_prefix = terminal(material_history(problem,
            np.vstack([fields[:first], fields[later]]), eta))
        case = dict(local_increment_mm=float(coordinates[later]),
            leg="unloading" if coordinates[later] < coordinates[later-1] else "reloading",
            history_state_counts=[first+1, later+1],
            first_pass=response(first), return_pass=response(later),
            return_minus_first=regional_contrast(host[rows], readout[rows]),
            observation_chain_contrast_microstrain=dict(
                uniform_finite_gauges=(sampled[later]-sampled[first]).tolist(),
                installed_linear_weights=(installed_linear[later]-installed_linear[first]).tolist(),
                finite_circuit_minus_installed_linear=(readout[later]-readout[first]
                    -installed_linear[later]+installed_linear[first]).tolist()),
            material_history=dict(first_pass=damage_summary(first_state, volume, area),
                return_pass=damage_summary(later_state, volume, area),
                adding_history_at_first_pass_field=prefix_effect(
                    first_state, first_with_later_prefix, volume, area),
                adding_history_at_return_pass_field=prefix_effect(
                    later_with_first_prefix, later_state, volume, area)))
        cases.append(case)
        print(json.dumps(dict(leg=case["leg"], local_mm=case["local_increment_mm"],
            host_contrast_rms=case["return_minus_first"]["host_rms_microstrain"],
            circuit_contrast_rms=case["return_minus_first"]["circuit_rms_microstrain"],
            history_gauge_max_microstrain=case["return_pass"]["history_gauge_abs_max_microstrain"])),
            flush=True)
    leg_states = []
    for index in [count-1]+[row["index"] for row in path_report["stages"]]:
        state = terminal(history, index)
        previous = terminal(history, index-1)
        leg_states.append(dict(**response(index),
            material_history=damage_summary(state, volume, area),
            bulk_damage_change_from_previous_volume_mean=float(
                (state["bulk_damage"]-previous["bulk_damage"])@volume/volume.sum()),
            interface_damage_change_from_previous_area_mean=float(
                (state["interface_damage"]-previous["interface_damage"])@area/area.sum())))
    report = dict(sources=dict(path=SOURCE.relative_to(ROOT).as_posix(),
            loading=source_path.relative_to(ROOT).as_posix(),
            circuit=DESIGN.relative_to(ROOT).as_posix(), checkpoint=path_report["checkpoint"]),
        role="Causal conditional local-control unloading and reloading from the saved loading prefix",
        contrast_order="return passage minus first passage at the same local displacement",
        axes=dict(directions=["1", "2"], regions=["left", "central", "right"],
                  components=["mean", "half_top_minus_bottom"], layers=["bottom", "top"]),
        nominal_material_parameters=eta, source_prefix_states=count, path_states=len(fields),
        source_prefix_preserved=True, causal_history_preserved=True,
        fixed_circuit_parameters=design,
        circuit_role="Installed low-load design evaluated unchanged on the conditional physical path",
        gauge_nominal_limit_microstrain=nominal_limit,
        complete_path_gauge_abs_max_microstrain=float(np.abs(raw).max()*1e6),
        complete_path_gauge_indices_above_nominal_limit=np.flatnonzero(
            np.abs(raw).max(axis=0)*1e6 > nominal_limit).tolist(),
        complete_path_amplifier_voltage_range_v=[float(circuit["voltage"].min()),
                                                  float(circuit["voltage"].max())],
        prefix_swap_role="Held endpoint fields isolate constitutive memory; these diagnostic swaps are not equilibrated paths",
        leg_states=leg_states, cases=cases)
    RESULT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    analyze()
