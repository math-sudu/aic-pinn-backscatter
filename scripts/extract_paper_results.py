"""Assemble retained paper quantities from the completed numerical artifacts.

This is deterministic result extraction. The saved producers remain the entry
points for field reconstruction, training, circuit selection and acquisition
calculations; this script does not perform those operations.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATERIAL = "results/ideation/stateful_pilot/fixed_reference_history"
SOURCES = {
    "physical_history": "results/ideation/B2_unloading_history_response.json",
    "control_physics": "results/ideation/B2_control_protocol_physics_response.json",
    "control_regional": "results/ideation/B2_control_protocol_regional_response.json",
    "control_design": "results/ideation/B2_control_protocol_design.json",
    "material_response": f"{MATERIAL}/regional_history_comparison.json",
    "material_design": "results/ideation/B2_history_design_value.json",
    "digital_readout": "results/ideation/B2_digital_readout.json",
    "wisp_readout": "results/ideation/B2_wisp_readout.json",
}
CONFIG_REFS = (
    "scripts/extract_paper_results.py",
    "inputs/control_history_protocols.json",
    "inputs/regional_readout.json",
    "inputs/digital_readout.json",
    "inputs/wisp_readout.json",
    "results/ideation/B2_control_protocol_physics.json",
    "results/ideation/B2_control_protocol_regional.json",
    f"{MATERIAL}/physics_history_metric.json",
    f"{MATERIAL}/physics_regional_control.json",
    f"{MATERIAL}/regional_history.json",
)
OUTPUT_REF = "results/paper/retained_results.json"


def fingerprint(refs):
    """Bind named text inputs in path order, independent of checkout newlines."""
    digest = hashlib.sha256()
    for ref in sorted(refs):
        digest.update(ref.encode("utf-8") + b"\0")
        content = (ROOT / ref).read_bytes().replace(b"\r\n", b"\n")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def reduction_percent(before, after):
    return 100 * (1 - after / before)


def assemble():
    records = {
        name: json.loads((ROOT / ref).read_text(encoding="utf-8"))
        for name, ref in SOURCES.items()
    }
    physical = records["physical_history"]
    design = records["control_design"]
    material = records["material_response"]
    material_design = records["material_design"]
    digital = records["digital_readout"]
    wisp = records["wisp_readout"]
    held_design = design["cross_evaluation"]["physical_evaluation"]["held"]
    initial = material["models"]["physics_history_metric"]["response_distance"]["held"]["discrepancy"]
    regional = material["models"]["regional_history"]["response_distance"]["held"]["discrepancy"]
    material_synthesis = {
        name: value["synthesis"]
        for name, value in material_design["circuits"].items()
    }
    return {
        "kind": "retained_paper_results",
        "operation": "Extract completed results and calculate named relative changes.",
        "sources": SOURCES,
        "config_refs": list(CONFIG_REFS),
        "config_hash": fingerprint(CONFIG_REFS),
        "data_fingerprint": fingerprint(SOURCES.values()),
        "control_history": {
            "claims": ["K1", "K2"],
            "axes": physical["axes"],
            "physical_contrasts": physical["cases"],
            "complete_path_gauge_abs_max_microstrain": physical["complete_path_gauge_abs_max_microstrain"],
            "predictions": {
                name: {
                    key: records[f"control_{name}"][key]
                    for key in ("training_protocols", "held_protocol", "protocol_change")
                }
                for name in ("physics", "regional")
            },
            "selection_controls_mm": design["training_controls_mm"],
            "held_controls_mm": design["held_controls_mm"],
            "designs": design["designs"],
            "cross_evaluation": design["cross_evaluation"],
            "physical_output_change": design["physical_output_change"],
            "held_design_composite_reduction_percent": reduction_percent(
                held_design["physics"]["objective_root_microstrain"],
                held_design["regional"]["objective_root_microstrain"],
            ),
            "comparison": "Regional versus physical-only selection, with the same starting network and two training protocols; assessment uses the excluded 0.75 mm same-model continuation.",
        },
        "material_history": {
            "claims": ["K2"],
            "material_parameters": material["material_parameters"],
            "force_kn": material["force_kn"],
            "groups": material["groups"],
            "models": material["models"],
            "design_synthesis": material_synthesis,
            "design_cross_evaluation": material_design["design_cross_evaluation"],
            "held_discrepancy_change_reduction_percent": reduction_percent(
                initial["material_change_rms_microstrain"],
                regional["material_change_rms_microstrain"],
            ),
            "held_centered_discrepancy_change_reduction_percent": reduction_percent(
                initial["material_change_components"]["centered_rms_microstrain"],
                regional["material_change_components"]["centered_rms_microstrain"],
            ),
            "held_design_composite_reduction_percent": reduction_percent(
                material_synthesis["physics_history_metric"]["held"]["objective_root_microstrain"],
                material_synthesis["regional_history"]["held"]["objective_root_microstrain"],
            ),
            "comparison": "The named reductions compare regional training/selection with the common starting network/circuit. Physical-only continuation remains a separate reported control. All candidates use the same local Newton-corrected assessment fields.",
        },
        "readout": {
            "claims": ["K3"],
            "designs": digital["designs"],
            "held_ranking": digital["held_ranking"],
            "regional_minus_physics": digital["regional_minus_physics"],
            "transport": digital["transport"],
            "scope": digital["source_scope"],
            "remaining_dependencies": digital["remaining_dependencies"],
            "wisp": {
                key: wisp[key]
                for key in ("stock_board", "designs", "epc_transmission_subtotal",
                            "receive_and_persistence_scaling", "scope")
            },
        },
    }


def main():
    result = assemble()
    output = ROOT / OUTPUT_REF
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT_REF, "config_hash": result["config_hash"],
                      "data_fingerprint": result["data_fingerprint"],
                      "control_design_reduction_percent": result["control_history"]["held_design_composite_reduction_percent"],
                      "material_design_reduction_percent": result["material_history"]["held_design_composite_reduction_percent"]}))


if __name__ == "__main__":
    main()
