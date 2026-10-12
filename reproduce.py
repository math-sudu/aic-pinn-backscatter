"""Run the paper's input, initialization, training and evaluation producers."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STAGES = {
    "inputs": [
        "fetch_jansson_data.py CL100H BL100H MaterialData",
        "analyze_jansson_observations.py",
        "analyze_jansson_mechanics.py",
        "analyze_engineering_alignment.py",
        "analyze_initial_state.py",
        "calibrate_support_response.py",
        "analyze_asymmetric_boundary.py",
        "analyze_boundary_history.py",
        "analyze_regional_material.py",
    ],
    "material": [
        "train_fixed_reference_history.py",
        "train_fixed_reference_history.py --resume --steps 36",
        "train_fixed_reference_history.py --name physics_absolute --initialize-from physics --path-coordinates absolute --steps 24",
        "train_fixed_reference_history.py --name physics_features --initialize-from physics_absolute --training-block features --steps 12",
        "train_fixed_reference_history.py --name physics_features_joint --initialize-from physics_features --training-block all --steps 24",
        "train_fixed_reference_history.py --name physics_material_span --initialize-from physics_features_joint --training-block all --width 8 --steps 12",
        "train_fixed_reference_history.py --name physics_material_coverage --initialize-from physics_material_span --material-grid-order 5 --training-block all --steps 24",
        "train_fixed_reference_history.py --name physics_history_continuation --initialize-from physics_material_coverage --steps 12",
        "train_fixed_reference_history.py --name physics_history_projected --initialize-from physics_history_continuation --optimizer projected_newton --steps 1 --workers 2",
        "train_fixed_reference_history.py --name physics_history_metric --initialize-from physics_history_projected --optimizer gauss_newton --linear-tolerance 0.02 --steps 2",
        "analyze_fixed_reference_history.py --name physics_history_metric",
        "train_fixed_reference_history.py --name physics_regional_control --initialize-from physics_history_metric --optimizer gauss_newton --linear-tolerance 0.02 --steps 1",
        "train_fixed_reference_history.py --name regional_history --initialize-from physics_history_metric --optimizer regional_history --readout-from physics_history_metric --linear-tolerance 0.02 --workers 2 --steps 1",
        "analyze_fixed_reference_history.py --name physics_regional_control --compare-to physics_history_metric --design-from physics_history_metric",
        "analyze_fixed_reference_history.py --name regional_history --compare-to physics_history_metric --design-from physics_history_metric",
        "analyze_regional_history_training.py --prepare-reference --workers 1",
        "analyze_regional_history_training.py",
        "analyze_history_design_value.py",
    ],
    "control_initialization": [
        "extend_loading_history.py",
        "train_controlled_fracture.py --name controlled_fracture --steps 12",
        "prepare_controlled_reference.py --steps 12",
        "train_controlled_fracture.py --name controlled_fracture_causal --resume --steps 3 --workers 2",
        "train_controlled_fracture.py --name controlled_fracture_balanced --initialize-from controlled_fracture_causal --balance-rigid --steps 6 --workers 2",
        "train_controlled_fracture.py --name controlled_fracture_balanced --resume --steps 12 --workers 2",
        "train_controlled_fracture.py --name controlled_fracture_body --initialize-from controlled_fracture_balanced --optimizer body_gauss_newton --steps 1",
        "train_controlled_fracture.py --name controlled_fracture_body --resume --optimizer body_gauss_newton --steps 3",
        "train_controlled_fracture.py --name controlled_fracture_body --resume --optimizer body_gauss_newton --steps 1",
        "train_controlled_fracture.py --name controlled_fracture_coupled --initialize-from controlled_fracture_body --optimizer full_gauss_newton --steps 1",
        "train_controlled_fracture.py --name controlled_fracture_equilibrated --initialize-from controlled_fracture_coupled --balance-constituents --optimizer full_gauss_newton --steps 0",
        "train_controlled_fracture.py --name controlled_fracture_implicit --initialize-from controlled_fracture_equilibrated --optimizer full_gauss_newton --steps 1",
        "train_controlled_fracture.py --name controlled_fracture_response --initialize-from controlled_fracture_implicit --optimizer response_gauss_newton --steps 1",
        "train_controlled_fracture.py --name controlled_fracture_amplitude --initialize-from controlled_fracture_response --optimizer amplitude_gauss_newton --steps 1 --workers 2",
        "train_controlled_fracture.py --name controlled_fracture_opening --initialize-from controlled_fracture_amplitude --optimizer opening_gauss_newton --steps 1",
        "train_controlled_fracture.py --name controlled_fracture_flexure --initialize-from controlled_fracture_opening --optimizer flexure_gauss_newton --steps 1",
        "analyze_controlled_continuation.py",
        "analyze_controlled_capacity.py --end-mm 10.436791840642528",
        "analyze_controlled_turning.py --local-targets-mm 0.4 0.45 0.5 0.6 --iterations 12",
        "analyze_controlled_turning.py --source results/ideation/B2_controlled_turning.json --local-targets-mm 0.8 --output results/ideation/B2_controlled_reachability.json --iterations 12",
        "analyze_controlled_turning.py --source results/ideation/B2_controlled_reachability.json --prefix-index 31 --local-targets-mm 0.6 0.8 --output results/ideation/B2_unloading_history.json --iterations 12",
        "train_control_history.py",
        "train_control_history.py --resume --optimizer gauss_newton --steps 2",
    ],
    "control": [
        "analyze_unloading_history_response.py",
        "analyze_controlled_turning.py --source results/ideation/B2_unloading_history.json --prefix-index 30 --local-targets-mm 0.7 0.6 --output results/ideation/B2_control_history_lower_peak.json --iterations 12",
        "analyze_controlled_turning.py --source results/ideation/B2_unloading_history.json --prefix-index 30 --local-targets-mm 0.75 0.6 --output results/ideation/B2_control_history_mid_peak.json --iterations 12",
        "train_control_history.py --initialize-from results/ideation/B2_control_history_training.json --protocols inputs/control_history_protocols.json --steps 1 --linear-iterations 120 --linear-tolerance 0.00001 --optimizer gauss_newton --output results/ideation/B2_control_protocol_physics.json",
        "train_control_history.py --initialize-from results/ideation/B2_control_history_training.json --protocols inputs/control_history_protocols.json --steps 1 --linear-iterations 120 --linear-tolerance 0.00001 --optimizer regional_history --output results/ideation/B2_control_protocol_regional.json",
        "analyze_control_protocols.py --checkpoint results/ideation/B2_control_protocol_physics.json --query-reference results/ideation/B2_control_history_mid_peak.json",
        "analyze_control_protocols.py --checkpoint results/ideation/B2_control_protocol_regional.json --query-reference results/ideation/B2_control_history_mid_peak.json",
        "analyze_control_protocol_design.py",
    ],
    "readout": [
        "analyze_digital_readout.py",
        "analyze_wisp_readout.py",
        "extract_paper_results.py",
    ],
}

PATH_RESULTS = {
    "analyze_controlled_continuation.py": ("results/ideation/B2_controlled_continuation.json", "recorded_path_completed"),
    "analyze_controlled_capacity.py": ("results/ideation/B2_controlled_capacity.json", "extended_path_completed"),
    "analyze_controlled_turning.py": ("results/ideation/B2_controlled_turning.json", "local_targets_completed"),
}


def command_args(command):
    script, *args = shlex.split(command)
    return [sys.executable, "-X", "utf8", str(ROOT / "scripts" / script), *args]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--list", action="store_true", help="Print commands without executing them")
    parser.add_argument("--start-at", type=int, default=1, help="First one-based step in the selected stage")
    args = parser.parse_args()
    if args.start_at < 1 or (args.stage == "all" and args.start_at != 1):
        parser.error("--start-at requires a positive index and one named stage")
    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"}
    for stage, commands in stages.items():
        for number, command in enumerate(commands, 1):
            if number < args.start_at:
                continue
            print(f"[{stage}:{number}] python scripts/{command}", flush=True)
            if args.list:
                continue
            subprocess.run(command_args(command), cwd=ROOT, env=env, check=True)
            tokens = shlex.split(command)
            if tokens[0] in PATH_RESULTS:
                ref, expected = PATH_RESULTS[tokens[0]]
                if "--output" in tokens:
                    ref = tokens[tokens.index("--output") + 1]
                actual = json.loads((ROOT / ref).read_text(encoding="utf-8"))["termination"]
                if actual != expected:
                    raise RuntimeError(
                        f"{ref}: {actual}; finish this path with the producer's --resume option "
                        f"before continuing at {stage}:{number + 1}"
                    )


if __name__ == "__main__":
    main()
