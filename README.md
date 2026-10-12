# Regional strain sensing with physics-informed learning

Source code for *Response-directed physics-informed learning for
history-dependent regional strain sensing* by Pengcheng Zhu and Tielin Chen.

The method learns regional host strains and fixed-circuit synthesis discrepancies
from causal physical corrections, then selects fixed resistor patterns from the
learned fields. The numerical study contains a control-history family and a
separate low-load material family. A component-level digital readout calculation
connects their fixed calibration to sequential acquisition and backscatter
snapshots.

## Installation and quick check

Use Python 3.11 or later and a CPU build of PyTorch. Install dependencies in a
virtual environment:

```console
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p test_digital_readout.py
python reproduce.py --list
```

The digital-readout tests run without downloaded data or trained networks. They
check the electrical node equations, reference calibration, acquisition charge
and snapshot encoding. The mechanics uses double precision and Gmsh-generated
quadratic tetrahedra. Its training and physical continuation stages require
substantial CPU time and memory; run them on a research workstation.

## Reproduce the numerical study

Run the following stages in order from this repository. `--list` prints the
exact commands in any stage without running them.

```console
python reproduce.py --stage inputs
python reproduce.py --stage material
python reproduce.py --stage control_initialization
python reproduce.py --stage control
python reproduce.py --stage readout
```

- `inputs` retrieves public specimen records, processes the observation and
  force histories, creates the mesh and constructs the low-load physical paths.
- `material` reconstructs the common starting material network, applies the
  physical-only and regional interventions, selects circuit candidates and
  evaluates them on a common local full-history correction.
- `control_initialization` constructs the supplied loading prefix and starting
  control-history network. The intermediate training steps preserve the
  initialization lineage used in the study.
- `control` trains on the 0.8 and 0.7 mm reversals, selects resistors on those
  neural fields, and evaluates the excluded 0.75 mm reversal. The path generator
  saves accepted states and supports `--resume` if a continuation stops before
  reaching its requested endpoint.
- `readout` computes the digital interface and charge scenarios, then writes
  `results/paper/retained_results.json`, the numerical values used in the paper.

The driver checks completion of each physical path before moving downstream.
After finishing an interrupted producer with its printed command and `--resume`,
continue with `python reproduce.py --stage STAGE --start-at NEXT_STEP`.
The included quick check needs only the input configurations.

The installed training circuits differ between the two families. The material
stage first runs `analyze_fixed_reference_history.py --name physics_history_metric`;
its `results/ideation/stateful_pilot/fixed_reference_history/physics_history_metric_readout.json`
records the material training pattern at `designs.selected.parameters`.
The regional training checkpoint copies those 16 values to
`setup.regional_history_training.circuit_parameters`. The control family uses
`designs.selected.parameters` from the same directory's
`regional_history_readout.json`, generated after regional material training.
Each vector contains four unit-sum groups: five left-region and three symmetric
central-region parameters for the x direction, then the same groups for y.
Right-region weights mirror the left; both depths share each pattern.
`scripts/weighted_readout.py:weight_map` expands these parameters into the
64-gauge layout. The final extraction also records the control training pattern
at `control_history.designs.installed.parameters` in `retained_results.json`.

The control and material networks have different assessment constructions.
The physical assessment is a calculation on the declared mechanics. The
specimen's measured support reload and high-load depth-strain allocation have
not been reproduced. The electronics results use the stated component noise,
installation and supply assumptions.

## Main implementation files

| File | Role |
| --- | --- |
| `scripts/control_history_network.py` | Causal shared fields and constituent/local-control layer |
| `scripts/control_history_response_training.py` | Causal physical corrections and response targets |
| `scripts/regional_history_training.py` | Finite-material response training with a learned reference |
| `scripts/weighted_readout.py` | Fixed shunt patterns and calibrated nonlinear circuit |
| `scripts/digital_readout.py` | Switched ratiometric interface, timing, charge and snapshot codec |
| `inputs/` | Mechanical, training and component configurations |

## Public observation data

`scripts/fetch_jansson_data.py` downloads the numerical members of the public
specimen archives from the [Swedish National Data Service collection](https://doi.org/10.5878/dvcn-bg03)
(version 1, CC0-1.0).
The archive's photographs and point clouds are not required by this code.
Please cite Jansson et al., *Experimental dataset for loads on hard rock
shotcrete tunnel linings in a laboratory environment*, Data in Brief 57 (2024),
[110920](https://doi.org/10.1016/j.dib.2024.110920), and observe the terms supplied
by the data repository.

This repository contains source, configurations and data-generation commands.
Downloaded observations, generated meshes, trained weights, numerical results,
plots and manuscript files are produced or maintained outside the published
source package.
