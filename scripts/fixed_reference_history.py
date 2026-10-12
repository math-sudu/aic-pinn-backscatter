"""Shared material histories with an explicit loaded reference and full forces."""

import copy
import itertools
import json

import numpy as np
import torch
from torch import nn

from analyze_asymmetric_boundary import load_problem
from analyze_spatial_transfer import raw_gauges
from boundary_history import ForceHistoryProblem
from functional_readout import PathLinearization, ReadoutFunctionals
from sparse_elastic_coordinates import SpatialCoordinates
from stateful_physics import ROOT, assemble_stiffness


OUTPUT = ROOT/"results/ideation/stateful_pilot/fixed_reference_history"


def material_training_grid(order):
    """Tensor Lobatto nodes retain the material endpoints and resolve the interior."""
    if order < 2:
        raise ValueError("A material axis needs at least two nodes")
    axis = -np.cos(np.linspace(0., np.pi, order))
    axis[np.abs(axis) < 1e-14] = 0.
    return np.asarray(list(itertools.product(axis, repeat=2)))


def parameter_indices(available, requested):
    """Match named material conditions, without assuming checkpoint array order."""
    available, requested = np.asarray(available), np.asarray(requested)
    matches = np.all(np.isclose(requested[:, None], available[None], rtol=0, atol=1e-12), axis=-1)
    if np.any(matches.sum(axis=1) != 1):
        raise ValueError("Each requested material condition must occur exactly once")
    return matches.argmax(axis=1)


def sparse_tensor(matrix, physics):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(physics.tensor(np.stack([coo.row, coo.col]), torch.long),
                                  physics.tensor(coo.data), coo.shape).coalesce()


def fixed_reference_components(absolute):
    """Store u0 and u_n-u0; this is an invertible path coordinate change."""
    return torch.cat([absolute[:, :1], absolute[:, 1:]-absolute[:, :1]], dim=1)


def absolute_components(reference_and_increments):
    return torch.cat([reference_and_increments[:, :1],
                      reference_and_increments[:, 1:]+reference_and_increments[:, :1]], dim=1)


class FixedReferenceNetwork(nn.Module):
    def __init__(self, coordinates, initial, width, path_coordinates="reference_increments"):
        super().__init__()
        self.coordinates = coordinates
        self.path_coordinates = path_coordinates
        p = coordinates.physics
        self.shape = (len(initial), sum(coordinates.sizes))
        self.layers = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, width),
                                    nn.Tanh(), nn.Linear(width, int(np.prod(self.shape))))
        self.register_buffer("feature_transform", torch.eye(width+1))
        self.to(device=p.device, dtype=p.dtype)
        with torch.no_grad():
            encoded = p.tensor(coordinates.encode_nominal(initial))[None]
            self.layers[-1].weight.zero_()
            raw = fixed_reference_components(encoded) if path_coordinates == "reference_increments" else encoded
            self.layers[-1].bias.copy_(raw.ravel())

    def features(self, parameters):
        hidden = self.layers[:-1](parameters)
        return torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=-1)

    def raw(self, parameters):
        last = self.layers[-1]
        coefficients = torch.cat([last.weight, last.bias[:, None]], dim=-1)
        return ((self.features(parameters)@self.feature_transform)@coefficients.T).reshape(
            len(parameters), *self.shape)

    def forward(self, parameters):
        raw = self.raw(parameters)
        if self.path_coordinates == "reference_increments":
            raw = absolute_components(raw)
        return self.coordinates.decode(raw)*self.coordinates.physics.scale

    @torch.no_grad()
    def expand_features(self, width):
        """Embed the old function and add independent, initially zero-output features."""
        old = self.layers
        previous_width = old[0].out_features
        if width <= previous_width:
            raise ValueError("Feature expansion requires a larger hidden width")
        p = self.coordinates.physics
        expanded = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, width),
                                 nn.Tanh(), nn.Linear(width, int(np.prod(self.shape))))
        expanded.to(device=p.device, dtype=p.dtype)
        expanded[0].weight[:previous_width].copy_(old[0].weight)
        expanded[0].bias[:previous_width].copy_(old[0].bias)
        expanded[2].weight[:previous_width].zero_()
        expanded[2].weight[:previous_width, :previous_width].copy_(old[2].weight)
        expanded[2].bias[:previous_width].copy_(old[2].bias)
        # Absorb the existing feature coordinates before extending the basis.
        coefficient = torch.cat([old[-1].weight, old[-1].bias[:, None]], dim=-1)@self.feature_transform.T
        expanded[-1].weight.zero_()
        expanded[-1].weight[:, :previous_width].copy_(coefficient[:, :-1])
        expanded[-1].bias.copy_(coefficient[:, -1])
        self.layers = expanded
        self.feature_transform = torch.eye(width+1, device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def reparameterize_path(self, kind):
        """Change time coordinates without changing any predicted displacement."""
        if kind not in ("absolute", "reference_increments"):
            raise ValueError(f"Unknown path coordinates: {kind}")
        if kind == self.path_coordinates:
            return
        last = self.layers[-1]
        coefficient = torch.cat([last.weight, last.bias[:, None]], dim=-1).reshape(*self.shape, -1)
        by_feature = coefficient.permute(2, 0, 1)
        transformed = (absolute_components(by_feature) if kind == "absolute"
                       else fixed_reference_components(by_feature))
        values = transformed.permute(1, 2, 0).reshape(-1, last.in_features+1)
        last.weight.copy_(values[:, :-1])
        last.bias.copy_(values[:, -1])
        self.path_coordinates = kind

    @torch.no_grad()
    def extend_path(self, state_map):
        """Initialize additional physical steps without tying their later updates.

        Rows map old absolute states to new absolute states. The original prefix
        must remain unchanged; repeated forces still get distinct output rows.
        The map is initialization only, never a constraint or a response label.
        """
        p = self.coordinates.physics
        mapping = p.tensor(state_map)
        old_steps = self.shape[0]
        if (mapping.ndim != 2 or mapping.shape[1] != old_steps
                or mapping.shape[0] <= old_steps):
            raise ValueError("A path extension must add steps to the existing path")
        torch.testing.assert_close(mapping[:old_steps], torch.eye(old_steps, device=p.device, dtype=p.dtype))
        torch.testing.assert_close(mapping.sum(dim=1), torch.ones(len(mapping), device=p.device, dtype=p.dtype))
        last = self.layers[-1]
        coefficients = torch.cat([last.weight, last.bias[:, None]], dim=-1)
        by_feature = coefficients.reshape(*self.shape, -1).permute(2, 0, 1)
        absolute = (absolute_components(by_feature) if self.path_coordinates == "reference_increments"
                    else by_feature)
        extended = torch.einsum("st,ftd->fsd", mapping, absolute)
        raw = (fixed_reference_components(extended) if self.path_coordinates == "reference_increments"
               else extended)
        self.shape = (len(mapping), self.shape[1])
        layer = nn.Linear(last.in_features, int(np.prod(self.shape))).to(device=p.device, dtype=p.dtype)
        values = raw.permute(1, 2, 0).reshape(-1, last.in_features+1)
        layer.weight.copy_(values[:, :-1])
        layer.bias.copy_(values[:, -1])
        self.layers[-1] = layer

    @torch.no_grad()
    def condition_features(self, parameters):
        features = self.features(parameters)
        _, values, vt = torch.linalg.svd(features/np.sqrt(len(parameters)), full_matrices=False)
        if len(values) != features.shape[1] or values[-1] <= torch.finfo(values.dtype).eps*values[0]*max(features.shape):
            raise ValueError("Material queries do not constrain the network features")
        last = self.layers[-1]
        coefficient = torch.cat([last.weight, last.bias[:, None]], dim=-1)
        transformed = (coefficient@self.feature_transform.T@vt.T)*values
        self.feature_transform.copy_(vt.T/values)
        last.weight.copy_(transformed[:, :-1])
        last.bias.copy_(transformed[:, -1])
        return values.cpu().tolist()


class MaterialHistoryProblem:
    def __init__(self, setup):
        self.setup = setup
        material = json.loads((ROOT/setup["material_configuration"]).read_text())
        boundary = json.loads((ROOT/material["boundary_configuration"]).read_text())
        asymmetric = json.loads((ROOT/boundary["asymmetric_configuration"]).read_text())
        report = json.loads((ROOT/boundary["asymmetric_report"]).read_text())
        source = next(row for row in report["cases"] if row["name"] == boundary["reference_case"])
        with np.load(ROOT/source["arrays"]) as saved:
            jack = saved["jack_operator"].copy()
        self.boundary_path = setup.get("boundary_arrays", material["boundary_arrays"])
        self.boundary = dict(np.load(ROOT/self.boundary_path))
        p, _, _ = load_problem(asymmetric)
        self.physics = p
        p.jack_operator = p.tensor(jack)
        self.initial = self.boundary["displacement_m"]
        self.replay = ForceHistoryProblem(p, self.initial[0], asymmetric["conditional_rod_stiffness_n_per_m"])
        self.replay.set_boundary(self.boundary["rod_force_n"], self.boundary["support_load_n"])
        self.force = p.tensor(self.boundary["force_n"])[None]
        # Subtracting the zero-top matrix leaves only that body's elastic force;
        # substrate, cone and lateral guides are identical in both matrices.
        settings = copy.deepcopy(p.settings)
        settings["bulk"]["top_young_pa"] = 0.
        zero_top = assemble_stiffness(p.mesh, settings)[0]
        self.top_stiffness_matrix = p._bulk_matrix-zero_top
        self.top_stiffness = sparse_tensor(self.top_stiffness_matrix, p)
        self.domain = p.tensor([setup["domain"][name] for name in
                               ("top_young_pa", "compression_stiffness_pa_per_m")])
        self.functionals = ReadoutFunctionals(p)
        self.raw_matrix = p.tensor(raw_gauges(p)[0][0])
        self.coordinates = SpatialCoordinates(p, parities=list(itertools.product((1, -1), repeat=2)),
                                               metric=self.replay.intact_tangent(self.initial[0]))

    def materials(self, parameters):
        return self.domain[:, 0]+(parameters+1)*.5*(self.domain[:, 1]-self.domain[:, 0])

    def equilibrium(self, displacement, parameters):
        material = self.materials(parameters)
        force, damage, history = self.replay.equilibrium(displacement, self.force,
                                                        compression_stiffness=material[:, 1, None, None])
        elastic = torch.sparse.mm(self.top_stiffness, displacement.reshape(-1, self.physics.ndof).T
                                  ).T.reshape(displacement.shape)
        ratio = material[:, 0]/self.physics.settings["bulk"]["top_young_pa"]-1
        return force+ratio[:, None, None]*elastic, damage, history

    def residual(self, network, parameters):
        return self.coordinates.whiten(self.equilibrium(network(parameters), parameters)[0])

    def linearization(self, displacement, parameters, *, projection=None):
        """Freeze the complete material-force Jacobian, including earlier history."""
        p = self.physics
        material = self.materials(p.tensor(parameters)).detach().cpu().numpy()
        ratio = material[0]/p.settings["bulk"]["top_young_pa"]-1
        bulk = p._bulk_matrix+ratio*self.top_stiffness_matrix+self.replay.gauge_matrix
        return PathLinearization(p, displacement, projection=projection, factorize=False,
                                 bulk_matrix=bulk, compression_stiffness=float(material[1]))

    def responses(self, displacement, readout):
        increments = displacement-displacement[:, :1]
        q = (increments@self.functionals.q_matrix.T).reshape(*displacement.shape[:2], 2, 3, 2)
        raw = displacement@self.raw_matrix.T
        absolute = readout(raw)
        return q, absolute-absolute[:, :1], raw

    def accumulate_loss(self, network, parameters, batch_size, diagnostics=None):
        total = 0.
        for start in range(0, len(parameters), batch_size):
            values = parameters[start:start+batch_size]
            force, damage, history = self.equilibrium(network(values), values)
            residual = self.coordinates.whiten(force)
            loss = residual.square().mean()*len(values)/len(parameters)
            if diagnostics is not None:
                diagnostics.extend(dict(parameters=eta.tolist(),
                    force_mean_square_by_step=mse.tolist(),
                    maximum_history_micro_m=peak.tolist(), maximum_damage=d.tolist())
                    for eta, mse, peak, d in zip(values.detach(), residual.detach().square().mean(dim=-1),
                        history.detach().amax(dim=-1)*1e6, damage.detach().amax(dim=-1)))
            loss.backward()
            total += float(loss.detach())
        return parameters.new_tensor(total)

    @torch.no_grad()
    def evaluate(self, network, parameters, readout):
        rows = []
        for parameter in parameters:
            eta = self.physics.tensor(parameter)[None]
            displacement = network(eta)
            q, circuit, raw = self.responses(displacement, readout)
            _, damage, history = self.equilibrium(displacement, eta)
            rows.append(dict(host_increment=q[0].numpy(), circuit_increment=circuit[0].numpy(),
                             raw_gauge_strain=raw[0].numpy(), maximum_damage=damage[0].amax(dim=-1).numpy(),
                             maximum_history_m=history[0].amax(dim=-1).numpy()))
        return dict(parameters=np.asarray(parameters), force_n=self.force[0].numpy(),
                    **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
