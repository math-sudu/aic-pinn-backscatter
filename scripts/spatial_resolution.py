"""Reflection-preserving coordinates for the symmetric nominal resolution test."""

import json

import numpy as np
from scipy import sparse
import torch
from torch import nn

from build_stateful_mesh import build_mesh
from quadratic_elements import upgrade_mesh
from stateful_physics import ROOT, PathPhysics

OUTPUT = ROOT/"results/ideation/stateful_pilot/spatial_resolution"


def reflection_projection(p, parity=(1, 1)):
    """Orthonormal coordinates for one vector-reflection parity sector.

    Reflection changes the normal vector component's sign in addition to the
    requested scalar parity. The four sectors together span every free DOF.
    Only operators at a reflection-symmetric state decouple into these sectors.
    """
    xyz = p.mesh["xyz"]
    body = np.empty(len(xyz), int)
    for b in range(3):
        body[np.unique(p.mesh["tetra"][p.mesh["body"] == b])] = b
    groups, rows, cols, signs = {}, [], [], []
    for row, dof in enumerate(p.free.cpu().numpy()):
        node, component = divmod(int(dof), 3)
        point = xyz[node]
        reflection_sign = [parity[axis]*(-1 if component == axis else 1) for axis in (0, 1)]
        if any(abs(point[axis]) < 1e-11 and reflection_sign[axis] == -1 for axis in (0, 1)):
            continue
        key = (int(body[node]), *np.round([abs(point[0]), abs(point[1]), point[2]], 11), component)
        col = groups.setdefault(key, len(groups))
        rows.append(row)
        cols.append(col)
        signs.append(np.prod([reflection_sign[axis] if point[axis] < 0 else 1 for axis in (0, 1)]))
    count = np.bincount(cols)
    values = np.asarray(signs)/np.sqrt(count[cols])
    return sparse.coo_matrix((values, (rows, cols)), shape=(len(p.free), len(groups))).tocsr()


def symmetric_projection(p):
    """The even sector used for the symmetric nominal field and loading."""
    return reflection_projection(p)


class SymmetricCoordinates:
    def __init__(self, physics):
        p = self.physics = physics
        self.projection = symmetric_projection(p)
        coo = self.projection.tocoo()
        self.lift = torch.sparse_coo_tensor(p.tensor(np.stack([coo.row, coo.col]), torch.long),
            p.tensor(coo.data), coo.shape, device=p.device).coalesce()
        free = p.free.cpu().numpy()
        matrix = p.elastic_stiffness()[free][:, free]
        reduced = (self.projection.T@matrix@self.projection).toarray()
        self.factor = torch.linalg.cholesky(p.tensor(reduced))
        obs = p.settings["observations"]["regions_m"]
        span = max(r[1] for r in obs)-min(r[0] for r in obs)
        self.coordinate_scale = np.sqrt(p.settings["bulk"]["top_young_pa"]*span)

    def expand(self, values):
        shape = values.shape
        result = torch.sparse.mm(self.lift, values.flatten(0, 1).T).T
        return result.reshape(*shape[:-1], self.lift.shape[0])

    def restrict(self, values):
        shape = values.shape
        result = torch.sparse.mm(self.lift.T, values.flatten(0, 1).T).T
        return result.reshape(*shape[:-1], self.lift.shape[1])

    def whiten(self, free_force):
        reduced = self.restrict(free_force)
        whitened = torch.linalg.solve_triangular(self.factor, reduced.flatten(0, 1).T, upper=False)
        return whitened.T.reshape(reduced.shape)/(self.coordinate_scale*self.physics.scale)


class ResolutionNetwork(nn.Module):
    def __init__(self, coordinates, width=32):
        super().__init__()
        self.coordinates = coordinates
        p = coordinates.physics
        self.nstep = len(p.path)
        self.nreduced = coordinates.lift.shape[1]
        self.layers = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, self.nstep*self.nreduced))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        self.to(device=p.device, dtype=p.dtype)

    def forward(self, amplitude):
        raw = self.layers((amplitude-1)[:, None]).reshape(-1, self.nstep, self.nreduced)
        raw = raw*amplitude[:, None, None]
        values = torch.linalg.solve_triangular(self.coordinates.factor.T, raw.flatten(0, 1).T, upper=True)
        return self.coordinates.expand(values.T.reshape(raw.shape))*self.coordinates.coordinate_scale


def load_resolution_problem(mesh_size, order=2, device="cpu", local_size=None):
    resolution = json.loads((ROOT/"inputs/spatial_resolution.json").read_text())
    settings = json.loads((ROOT/resolution["base_settings"]).read_text())
    spatial = json.loads((ROOT/resolution["support_settings"]).read_text())
    settings["geometry"]["mesh_size_m"] = mesh_size
    settings["geometry"]["lateral_guide_stiffness_n_per_m"] = spatial["lateral_guide_stiffness_n_per_m"]
    suffix = "" if mesh_size == 0.15 else "_"+f"{mesh_size:.5f}".replace(".", "p")
    if local_size is not None:
        extent = max(abs(value) for row in settings["observations"]["regions_m"] for value in row)
        settings["geometry"]["readout_resolution"] = dict(size_m=local_size, extent_m=extent,
            transition_m=resolution["transition_m"])
        suffix += "_readout_"+f"{local_size:.5f}".replace(".", "p")
    path = ROOT/f"data/processed/stateful_pilot/symmetric_mesh{suffix}.npz"
    if path.exists():
        with np.load(path) as source:
            mesh = dict(source)
    else:
        mesh = build_mesh(settings, reflection_symmetric=True)
        np.savez_compressed(path, **mesh)
    if order == 2:
        mesh = upgrade_mesh(mesh, settings)
    return PathPhysics(mesh, settings, device)
