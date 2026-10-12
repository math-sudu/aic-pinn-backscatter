"""One parameterized neural history over the prescribed three-shape bond domain."""

import itertools
import json

import numpy as np
import torch
from torch import nn

from analyze_spatial_transfer import bond_basis, raw_gauges
from functional_readout import ReadoutFunctionals
from sparse_elastic_coordinates import SpatialCoordinates
from spatial_resolution import OUTPUT as SOURCE, load_resolution_problem
from stateful_physics import ROOT


OUTPUT = ROOT/"results/ideation/stateful_pilot/shared_spatial_surrogate"


def sample_cases(shapes, kind):
    labels, rows = ["nominal"], [np.zeros(3)]
    for j, name in enumerate(shapes):
        for sign in (-1, 1):
            row = np.zeros(3)
            row[j] = sign
            labels.append(f"{name}_{sign:+g}")
            rows.append(row)
    if kind in ("train", "coverage"):
        groups = [("face", np.full(3, 1/3)), ("interior", np.full(3, 1/6))]
        if kind == "coverage":
            groups += [(f"boundary_{j}", np.asarray(values)) for j, values in enumerate(
                sorted(set(itertools.permutations((.5, .25, .25)))))]
    elif kind == "query":
        groups = [("mixed_boundary", np.array([.5, .3, .2])),
                  ("mixed_interior", np.array([.2, .1, .3]))]
    else:
        raise ValueError(f"Unknown sample role: {kind}")
    for name, magnitudes in groups:
        for signs in itertools.product((-1, 1), repeat=3):
            labels.append(name+"_"+"".join("p" if sign > 0 else "m" for sign in signs))
            rows.append(magnitudes*np.array(signs))
    return np.asarray(labels), np.stack(rows)


def reflection_representatives(parameters):
    """Equal-weight full-domain training via exact x-reflection pairs."""
    reflected = parameters.copy()
    reflected[:, 1] = abs(reflected[:, 1])
    unique, first, counts = np.unique(reflected, axis=0, return_index=True, return_counts=True)
    order = np.argsort(first)
    return unique[order], counts[order]/len(parameters)


class SharedSpatialNetwork(nn.Module):
    def __init__(self, coordinates, width, initial):
        super().__init__()
        self.coordinates = coordinates
        p = coordinates.physics
        self.shape = (len(p.path), sum(coordinates.sizes))
        self.layers = nn.Sequential(nn.Linear(3, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, int(np.prod(self.shape))))
        self.register_buffer("feature_transforms", torch.eye(width+1).repeat(len(coordinates.sizes), 1, 1))
        self.to(device=p.device, dtype=p.dtype)
        with torch.no_grad():
            self.layers[-1].weight.zero_()
            self.layers[-1].bias.copy_(p.tensor(coordinates.encode_nominal(initial).ravel()))

    def features(self, eta):
        features = torch.stack([eta[:, 0], eta[:, 1].square(), eta[:, 2]], dim=-1)
        hidden = self.layers[:-1](features)
        return torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=-1)

    def coefficients(self):
        last = self.layers[-1]
        return torch.cat([last.weight, last.bias[:, None]], dim=-1).reshape(*self.shape, -1)

    def raw(self, eta):
        features = self.features(eta)
        coefficients = torch.split(self.coefficients(), self.coordinates.sizes, dim=1)
        sectors = []
        for transform, coefficient, parity in zip(self.feature_transforms, coefficients,
                                                   self.coordinates.parities):
            sector = torch.einsum("bf,sdf->bsd", features@transform, coefficient)
            sectors.append(sector if parity[0] == 1 else sector*eta[:, 1, None, None])
        return torch.cat(sectors, dim=-1)

    @torch.no_grad()
    def condition_features(self, eta, weights):
        """Whiten each sector's sampled features without changing its function.

        For A = sqrt(w) g(eta) h(eta) = U S V.T, use h V S^-1
        and transform the output coefficients by V S. Hidden layers remain
        trainable; the transformation is fixed until the next training segment.
        """
        features = self.features(eta)
        coefficients = torch.split(self.coefficients(), self.coordinates.sizes, dim=1)
        transformed, transforms, spectra = [], [], []
        for i, (coefficient, parity) in enumerate(zip(coefficients, self.coordinates.parities)):
            factor = torch.ones_like(eta[:, 1]) if parity[0] == 1 else eta[:, 1]
            matrix = features*(weights.sqrt()*factor)[:, None]
            _, values, vt = torch.linalg.svd(matrix, full_matrices=False)
            if len(values) != features.shape[1] or values[-1] <= (
                    torch.finfo(values.dtype).eps*max(matrix.shape)*values[0]):
                raise ValueError("Training conditions do not constrain every sector feature")
            physical = coefficient@self.feature_transforms[i].T
            transformed.append((physical@vt.T)*values)
            transforms.append(vt.T/values)
            spectra.append(dict(parity=parity, singular_values=values.cpu().tolist(),
                active_conditions=int(torch.count_nonzero(factor))))
        last = torch.cat(transformed, dim=1).reshape(-1, features.shape[1])
        self.feature_transforms.copy_(torch.stack(transforms))
        self.layers[-1].weight.copy_(last[:, :-1])
        self.layers[-1].bias.copy_(last[:, -1])
        return spectra

    def forward(self, eta):
        return self.coordinates.decode(self.raw(eta))


def load_network(network, path, device):
    state = torch.load(path, map_location=device, weights_only=True)["network"]
    # The saved physics baseline predates feature coordinates and uses identity.
    state.setdefault("feature_transforms", torch.eye(network.layers[-1].in_features+1,
        device=network.feature_transforms.device, dtype=network.feature_transforms.dtype
        ).repeat(len(network.coordinates.sizes), 1, 1))
    network.load_state_dict(state)


class SharedProblem:
    def __init__(self, setup, device):
        self.setup = setup
        source = json.loads((SOURCE/f"{setup['reference']}.json").read_text())
        p = self.physics = load_resolution_problem(source["mesh_size_m"], source["order"],
                                                   device, source["local_size_m"])
        self.basis_settings = json.loads((ROOT/"inputs/spatial_transfer.json").read_text())
        basis, names, _ = bond_basis(p, self.basis_settings)
        self.basis = p.tensor(basis[:, [names.index(name) for name in setup["shapes"]]])
        _, endpoints = sample_cases(setup["shapes"], "train")
        scale = self.bond_scale(p.tensor(endpoints)).cpu().numpy()
        self.coordinates = SpatialCoordinates(p, scale)
        self.functionals = ReadoutFunctionals(p)
        self.raw_matrix = p.tensor(raw_gauges(p)[0][0])
        with np.load(SOURCE/f"{setup['reference']}.npz") as saved:
            self.initial = saved["displacement_m"][0]

    def bond_scale(self, eta):
        return (1+self.basis_settings["bond_relative_scale"]*(eta@self.basis.T))[:, None]

    def displacement(self, network, eta):
        return self.physics.displacements(network(eta), eta.new_ones(len(eta)))

    def residual(self, network, eta):
        return self.coordinates.whiten(self.force(network, eta))

    def force(self, network, eta):
        p = self.physics
        return p.forces(self.displacement(network, eta), bond_scale=self.bond_scale(eta))[0][:, :, p.free]

    def force_tangent(self, network, eta, axis, create_graph=False):
        direction = torch.zeros_like(eta)
        direction[:, axis] = 1
        return torch.autograd.functional.jvp(lambda value: self.force(network, value),
            eta, direction, create_graph=create_graph)

    def residual_tangent(self, network, eta, axis, create_graph=False):
        """Total parameter derivative, including explicit bond and prior history."""
        direction = torch.zeros_like(eta)
        direction[:, axis] = 1
        return torch.autograd.functional.jvp(lambda value: self.residual(network, value),
            eta, direction, create_graph=create_graph)

    def responses(self, network, eta, readout):
        """Host and installed-circuit responses; both retain their own state map."""
        u = self.displacement(network, eta)
        q = (u@self.functionals.q_matrix.T).reshape(*u.shape[:-1], 2, 3, 2)
        return torch.stack([q, readout(u@self.raw_matrix.T)], dim=-1)

    def accumulate_loss(self, network, parameters, weights, batch_size, sensitivity=False,
                        readout_metric=None):
        """Accumulate the exact full loss gradient without retaining every graph."""
        if readout_metric is not None:
            return readout_metric.accumulate_loss(self, network, parameters, weights)
        totals = dict(physics=0., sensitivity=[0., 0., 0.])
        for start in range(0, len(parameters), batch_size):
            eta = parameters[start:start+batch_size]
            batch_weights = weights[start:start+batch_size]

            def squared_loss(value):
                return (value.square().mean(dim=(1, 2))*batch_weights).sum()*1e5

            if sensitivity:
                for axis in range(3):
                    residual, tangent = self.residual_tangent(network, eta, axis, create_graph=True)
                    tangent_loss = squared_loss(tangent)
                    totals["sensitivity"][axis] += float(tangent_loss.detach())
                    loss = tangent_loss
                    if axis == 0:
                        physical_loss = squared_loss(residual)
                        totals["physics"] += float(physical_loss.detach())
                        loss = loss+physical_loss
                    loss.backward()
            else:
                loss = squared_loss(self.residual(network, eta))
                totals["physics"] += float(loss.detach())
                loss.backward()
        self.last_loss_terms = totals
        return parameters.new_tensor(totals["physics"]+sum(totals["sensitivity"]))

    def evaluate(self, network, labels, parameters, batch_size, include_fields=False):
        p, parts = self.physics, []
        with torch.no_grad():
            for start in range(0, len(parameters), batch_size):
                eta = p.tensor(parameters[start:start+batch_size])
                u = self.displacement(network, eta)
                q = torch.einsum("...d,od->...o", u, self.functionals.q_matrix).reshape(-1, len(p.path), 2, 3, 2)
                gauges = torch.einsum("od,bsd->bso", self.raw_matrix, u)
                row = dict(host_target=q, raw_gauge_strain=gauges)
                if include_fields:
                    scale = self.bond_scale(eta)
                    force = p.forces(u, bond_scale=scale)[0]
                    row.update(displacement_m=u, free_force_n=force[:, :, p.free], bond_scale=scale)
                parts.append({key: value.cpu().numpy() for key, value in row.items()})
        arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        arrays.update(case_labels=labels, parameters=parameters)
        return arrays
