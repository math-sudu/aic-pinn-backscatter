"""Finite spatial scenarios, symmetric mechanics and neural path coordinates."""

import json

import numpy as np
import torch
from torch import nn

from analyze_spatial_transfer import bond_basis
from build_stateful_mesh import build_mesh
from stateful_physics import ROOT, PathPhysics


OUTPUT = ROOT/"results/ideation/stateful_pilot/spatial_finite_paths"


def load_spatial_problem(device="cpu", mesh_size=None):
    settings = json.loads((ROOT/"inputs/spatial_finite_paths.json").read_text())
    physical = json.loads((ROOT/settings["base_settings"]).read_text())
    physical["geometry"]["lateral_guide_stiffness_n_per_m"] = settings["lateral_guide_stiffness_n_per_m"]
    if mesh_size is not None and mesh_size != physical["geometry"]["mesh_size_m"]:
        physical["geometry"]["mesh_size_m"] = mesh_size
        suffix = f"{mesh_size:.5f}".replace(".", "p")
        settings["mesh_path"] = f"data/processed/stateful_pilot/symmetric_mesh_{suffix}.npz"
    settings["mesh_size_m"] = physical["geometry"]["mesh_size_m"]
    path = ROOT/settings["mesh_path"]
    if not path.exists():
        mesh = build_mesh(physical, reflection_symmetric=True)
        np.savez_compressed(path, **mesh)
    else:
        with np.load(path) as source:
            mesh = dict(source)
    physics = PathPhysics(mesh, physical, device)
    basis_settings = json.loads((ROOT/settings["basis_settings"]).read_text())
    basis, names, _ = bond_basis(physics, basis_settings)
    return physics, settings, basis_settings, basis, names


def finite_cases(settings, names):
    labels, parameters = ["nominal"], [np.zeros(len(names)+1)]
    for shape in settings["representative_shapes"]:
        vector = np.zeros(len(names)+1)
        if shape == "double_odd_1_1":
            vector[1+names.index("bond_cos_1_-1")] = 0.5
            vector[1+names.index("bond_cos_1_1")] = -0.5
        else:
            vector[1+names.index(shape)] = 1
        for multiplier in settings["finite_multipliers"]:
            labels.append(f"{shape}_{multiplier:+g}")
            parameters.append(multiplier*vector)
    return labels, np.stack(parameters)


def reflection_maps(physics, axis):
    """Involutive node and interface-quadrature permutations for x or y."""
    mesh = physics.mesh
    body = np.empty(len(mesh["xyz"]), dtype=int)
    for b in range(3):
        body[np.unique(mesh["tetra"][mesh["body"] == b])] = b
    signs = np.ones(3)
    signs[axis] = -1
    lookup = {(int(b), *np.round(x, 11)): i for i, (b, x) in enumerate(zip(body, mesh["xyz"]))}
    nodes = np.asarray([lookup[(int(b), *np.round(x*signs, 11))]
                        for b, x in zip(body, mesh["xyz"])])
    dofs = (3*nodes[:, None]+np.arange(3)).ravel()
    point = np.einsum("qa,fac->fqc", physics.shape.cpu().numpy(),
                      mesh["xyz"][mesh["interface_upper"]]).reshape(-1, 3)
    lookup = {tuple(np.round(x, 11)): i for i, x in enumerate(point)}
    quadrature = np.asarray([lookup[tuple(np.round(x*signs, 11))] for x in point])
    return dofs, np.tile(signs, len(nodes)), quadrature


class SpatialPathNetwork(nn.Module):
    """Scaled loading/bond parameters -> complete path in fixed elastic coordinates."""

    def __init__(self, physics, width, dimension):
        super().__init__()
        self.nstep, self.nfree = len(physics.path), len(physics.free)
        self.layers = nn.Sequential(nn.Linear(dimension, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, self.nstep*self.nfree))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        self.factor = physics.elastic_metric
        obs = physics.settings["observations"]["regions_m"]
        span = max(r[1] for r in obs)-min(r[0] for r in obs)
        self.coordinate_scale = np.sqrt(physics.settings["bulk"]["top_young_pa"]*span)
        self.to(device=physics.device, dtype=physics.dtype)

    def forward(self, parameters, amplitudes):
        raw = self.layers(parameters).reshape(-1, self.nstep, self.nfree)*amplitudes[:, None, None]
        values = torch.linalg.solve_triangular(self.factor.T, raw.flatten(0,1).T, upper=True)
        return values.T.reshape(raw.shape)*self.coordinate_scale


def path_inputs(physics, parameters, basis_settings, basis):
    a = 1+basis_settings["amplitude_relative_scale"]*parameters[:, 0]
    scale = 1+basis_settings["bond_relative_scale"]*(parameters[:, 1:]@physics.tensor(basis).T)
    return a, scale[:, None]
