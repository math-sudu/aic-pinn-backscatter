"""Gravity seating, compliant anchors and local relative-motion observations.

The existing PathPhysics supplies the three-dimensional bulk/interface/contact
equations. This experimental-alignment model frees all vertical boundary DOFs,
applies one jack force resultant and retains the complete cohesive path.
"""

import json
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch

from jansson_materials import read_cube
from build_stateful_mesh import build_mesh
from quadratic_elements import TRI_EDGES, shape_values, surface_matrix, upgrade_mesh
from stateful_physics import PathPhysics, ROOT


def material_densities(config):
    records = {}
    for path in sorted((ROOT / "data/raw/jansson2024/MaterialData").rglob("*.tra")):
        metadata = read_cube(path)[0]
        density = float(metadata["Bulk density"][0].replace(",", ".")) * 1000
        mass = float(metadata["Weight of the specimen"][0].replace(",", ".")) / 1000
        volume = np.prod([float(metadata[key][0].replace(",", ".")) / 1000
                          for key in ("Specimen thickness a", "Specimen width b", "Specimen height h")])
        records[path.stem] = dict(density_kg_per_m3=density,
                                 mass_over_volume_kg_per_m3=float(mass / volume),
                                 source=path.relative_to(ROOT).as_posix())
    groups = {prefix: float(np.mean([row["density_kg_per_m3"] for name, row in records.items()
                                    if name.startswith(prefix)]))
              for prefix in [config["top_cube_prefix"], *config["substrate_cube_prefixes"]]}
    substrate = np.mean([row["density_kg_per_m3"] for name, row in records.items()
                         if any(name.startswith(prefix) for prefix in config["substrate_cube_prefixes"])])
    return np.array([groups[config["top_cube_prefix"]], substrate, substrate]), records, groups


def gravity_load(mesh, volumes, densities, gravity):
    """Consistent constant body load; P2 vertex weights are negative, not lumped."""
    weights = (np.full(4, 0.25) if mesh["tetra"].shape[1] == 4
               else np.r_[np.full(4, -0.05), np.full(6, 0.2)])
    force = np.zeros((len(mesh["xyz"]), 3))
    local = -volumes[:, None] * densities[mesh["body"], None] * gravity * weights
    np.add.at(force[:, 2], mesh["tetra"].ravel(), local.ravel())
    return force.ravel()


def surface_faces(mesh, body, level):
    """Boundary triangles at a horizontal level, including the P2 edge nodes."""
    tetra = mesh["tetra"][mesh["body"] == body]
    rows = []
    tet_edges = list(combinations(range(4), 2))
    for corners in combinations(range(4), 3):
        selected = tetra[np.all(np.isclose(mesh["xyz"][tetra[:, corners], 2], level), axis=1)]
        columns = list(corners)
        if tetra.shape[1] == 10:
            columns += [4 + tet_edges.index(tuple(sorted((corners[a], corners[b]))))
                        for a, b in TRI_EDGES]
        rows.append(selected[:, columns])
    faces = np.concatenate(rows)
    vertices = mesh["xyz"][faces[:, :3]]
    areas = np.linalg.norm(np.cross(vertices[:, 1] - vertices[:, 0],
                                   vertices[:, 2] - vertices[:, 0]), axis=1) / 2
    return faces, areas


def patch_mean(mesh, body, level, centre, radius):
    faces, area = surface_faces(mesh, body, level)
    selected = np.linalg.norm(mesh["xyz"][faces[:, :3], :2].mean(axis=1) - centre, axis=1) < radius
    faces, area = faces[selected], area[selected]
    if not len(faces):
        raise ValueError("The requested load/bearing patch has no boundary faces")
    weights = np.full(3, 1 / 3) if faces.shape[1] == 3 else np.r_[np.zeros(3), np.full(3, 1 / 3)]
    operator = np.zeros(3 * len(mesh["xyz"]))
    np.add.at(operator, 3 * faces.ravel() + 2, (area[:, None] * weights / area.sum()).ravel())
    return operator


def point_value(mesh, inverse, body, point):
    elements = np.flatnonzero(mesh["body"] == body)
    bary = np.einsum("i,eij->ej", np.r_[1, point], inverse[elements])
    valid = np.flatnonzero(np.min(bary, axis=1) >= -1e-9)
    if not len(valid):
        raise ValueError(f"Point {point} is outside body {body}")
    index = valid[0]
    weights = bary[index] if mesh["tetra"].shape[1] == 4 else shape_values(bary[index])
    operator = np.zeros(3 * len(mesh["xyz"]))
    operator[3 * mesh["tetra"][elements[index]] + 2] = weights
    return operator


def relative_edge_probes(mesh, inverse, angles_deg):
    """One-sided values at identical bottom-edge coordinates on the two bodies."""
    bottom = mesh["xyz"][mesh["wall_cone"], 2].min()
    edges = []
    for face in mesh["wall_cone"][:, :3]:
        nodes = face[np.isclose(mesh["xyz"][face, 2], bottom)]
        if len(nodes) == 2:
            edges.append(mesh["xyz"][nodes, :2])
    operators, points = [], []
    for angle in np.deg2rad(angles_deg):
        ray = np.array([np.cos(angle), np.sin(angle)])
        matches = []
        for a, b in edges:
            matrix = np.column_stack([ray, a - b])
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            radius, fraction = np.linalg.solve(matrix, a)
            if radius > 0 and -1e-9 <= fraction <= 1 + 1e-9:
                matches.append(radius)
        if not matches:
            raise ValueError("The probe ray does not intersect the cone bottom edge")
        point = np.r_[min(matches) * ray, bottom]
        operators.append(point_value(mesh, inverse, 2, point) - point_value(mesh, inverse, 1, point))
        points.append(point)
    return np.asarray(operators), np.asarray(points)


class InitialStatePhysics(PathPhysics):
    def __init__(self, mesh, settings, config, densities):
        mesh = dict(mesh, fixed_dofs=np.array([], dtype=int), driven_dofs=np.array([], dtype=int))
        super().__init__(mesh, settings)
        self.config = config
        self.gravity = self.tensor(gravity_load(mesh, self.volume, densities, config["weight"]["gravity_m_per_s2"]))
        geometry = settings["geometry"]
        self.anchor_operator = self.tensor(np.stack([
            patch_mean(mesh, 0, geometry["top_levels_m"][-1], centre, geometry["bolt_patch_radius_m"])
            for centre in geometry["bolt_centres_m"]]))
        self.anchor_stiffness = self.tensor(config["anchors"]["stiffness_n_per_m"])
        self.assembly_tension = self.tensor(config["anchors"]["assembly_tension_n"])
        self.anchor_reference = None
        self.jack_operator = self.tensor(patch_mean(mesh, 2, geometry["bottom_levels_m"][-1],
                                                   np.zeros(2), geometry["jack_patch_radius_m"]))
        relative, self.probe_points = relative_edge_probes(mesh, self.inverse,
                                                          config["local_observations"]["angles_deg"])
        self.relative_operator = self.tensor(relative)
        zero = np.zeros(self.ndof)
        self.intact_bulk_interface = (super().elastic_stiffness() - self.trestle_tangent(zero)
                                      - self.wall_tangent(zero)).tocsr()

    def anchor_loads(self, displacement):
        mean = torch.einsum("...d,ad->...a", displacement, self.anchor_operator)
        if self.anchor_reference is None:
            return self.assembly_tension.expand_as(mean)
        return (self.assembly_tension + self.anchor_stiffness * (mean - self.anchor_reference)).clamp_min(0)

    def equilibrium(self, displacement, jack_force_n):
        """Full physical-path residual, with SI force and displacement inputs."""
        force, damage, history = super().forces(displacement)
        anchors = torch.einsum("...a,ad->...d", self.anchor_loads(displacement), self.anchor_operator)
        residual = force + anchors - self.gravity - jack_force_n[..., None] * self.jack_operator
        return residual, damage, history

    def seat_anchors(self, equilibrium_displacement):
        # Recompute this state when gravity or assembly tensions change.
        self.anchor_reference = self.anchor_operator @ equilibrium_displacement

    def wall_tangent(self, displacement):
        mesh = self.mesh
        shape, weights = self.shape.numpy(), self.surface_weights.numpy()
        signed = np.concatenate([shape, -shape], axis=1)
        gap = self.wall_gap(self.tensor(displacement)[None, None])[0, 0].numpy()
        normal = mesh["wall_normal"]
        tangent = np.einsum("fq,fi,fj->fqij", gap <= 0, normal, normal)
        tangent *= self.settings["geometry"]["cone_contact_stiffness_pa_per_m"]
        width = 6 * shape.shape[1]
        local = np.einsum("qa,fqij,qb,f,q->faibj", signed, tangent, signed,
                          mesh["wall_area"], weights, optimize=True).reshape(-1, width, width)
        return surface_matrix(np.concatenate([mesh["wall_substrate"], mesh["wall_cone"]], axis=1),
                              local, self.ndof)

    def intact_tangent(self, displacement):
        matrix = (self.intact_bulk_interface + self.interface_closure_tangent(displacement)
                  + self.trestle_tangent(displacement) + self.wall_tangent(displacement))
        if self.anchor_reference is not None:
            operator = sparse.csr_matrix(self.anchor_operator.numpy())
            extension = (self.assembly_tension + self.anchor_stiffness *
                         (self.anchor_operator @ self.tensor(displacement) - self.anchor_reference)).numpy()
            matrix += operator.T @ sparse.diags(self.anchor_stiffness.numpy() * (extension >= 0)) @ operator
        return matrix.tocsc()

    def mechanical_outputs(self, displacement):
        nodal = displacement.reshape(*displacement.shape[:-1], -1, 3)
        gap = torch.einsum("qa,...fa->...fq", self.shape, nodal[..., self.trestle_faces, 2])
        trestle = -torch.einsum("...fq,f,q->...", gap.clamp_max(0), self.trestle_face_area,
                                self.surface_weights) * self.settings["geometry"]["trestle_stiffness_pa_per_m"]
        return dict(rod_force_n=self.anchor_loads(displacement), trestle_force_n=trestle,
                    local_relative_m=torch.einsum("...d,od->...o", displacement, self.relative_operator),
                    jack_face_mean_m=displacement @ self.jack_operator)


def solve_intact(physics, force_n, initial=None, projection=None, linear_solver=None):
    """Static initialization/low-load diagnostic only; reject material damage.

    The general nonlinear history is evaluated by equilibrium(), not this
    initializer. Its stopping tolerance is an algebraic solver setting.
    """
    p = physics
    current = np.zeros(p.ndof) if initial is None else np.array(initial, copy=True)
    for iteration in range(40):
        residual, damage, _ = p.equilibrium(p.tensor(current)[None, None], p.tensor([[force_n]]))
        if torch.any(damage > 0):
            raise ValueError("The intact initializer reached cohesive damage; use a nonlinear history solver")
        residual = residual.detach().numpy().ravel()
        if np.max(np.abs(residual)) < 1e-5:
            return current, iteration
        tangent = p.intact_tangent(current)
        if linear_solver is not None:
            step = linear_solver(tangent, -residual)
        elif projection is None:
            step = spsolve(tangent, -residual, permc_spec="MMD_AT_PLUS_A")
        else:
            reduced = (projection.T @ tangent @ projection).tocsc()
            step = projection @ spsolve(reduced, -projection.T @ residual, permc_spec="MMD_AT_PLUS_A")
        current += step
    raise RuntimeError("Initial contact/anchor equilibrium did not converge")


def build_initial_problem():
    config = json.loads((ROOT / "inputs/initial_state_model.json").read_text(encoding="utf-8"))
    settings = json.loads((ROOT / config["base_settings"]).read_text(encoding="utf-8"))
    settings["geometry"].update(config["geometry"])
    densities, records, groups = material_densities(config["weight"])
    mesh = upgrade_mesh(build_mesh(settings, reflection_symmetric=True), settings)
    return InitialStatePhysics(mesh, settings, config, densities), records, groups
