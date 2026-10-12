"""Shared constituent-motion updates using the actual reactive history equations.

The two relative rigid-body fields change interface/contact transfer without
changing bulk strain. Their coefficients use the existing material features
and are folded into the original displacement output layer after an update.
"""

import json

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
import torch



class BodyMotionBasis:
    def __init__(self, problem):
        p = self.physics = problem.physics
        xyz = p.mesh["xyz"]
        centre = np.array([0., 0., xyz[:, 2].min()])
        fields = np.zeros((len(xyz), 3, 12))
        # Substrate motion is supplied by the common rigid reaction layer.
        for index, body in enumerate((0, 2)):
            nodes = np.unique(p.mesh["tetra"][p.mesh["body"] == body])
            fields[nodes, :, 6*index:6*index+3] = np.eye(3)
            for axis in range(3):
                fields[nodes, :, 6*index+3+axis] = np.cross(np.eye(3)[axis], xyz[nodes]-centre)
        physical = fields.reshape(p.ndof, 12)
        encoded = problem.coordinates.encode_nominal(physical.T).T
        # Coordinates only: the actual tangent below supplies the update metric.
        root = np.linalg.cholesky(encoded.T@encoded)
        transform = np.linalg.solve(root.T, np.eye(12))
        self.centre, self.transform = centre, transform
        self.modes = p.tensor(physical@transform)
        self.encoded = p.tensor(encoded@transform)
        nodal = self.modes.reshape(-1, 3, 12)
        self.jump_modes = torch.einsum("qa,fack->fqck", p.shape, nodal[p.upper]-nodal[p.lower])
        self.wall_modes = torch.einsum("qa,fack,fc->fqk", p.shape,
            nodal[p.wall_substrate]-nodal[p.wall_cone], p.wall_normal)
        self.support_modes = torch.einsum("qa,fak->fqk", p.shape, nodal[p.trestle_faces, 2])
        self.guide = p.tensor(p.mesh["lateral_guide_operator"])
        self.guide_modes = self.guide@self.modes
        self.anchor_modes = p.anchor_operator@self.modes
        self.jack_modes = p.jack_operator@self.modes
        self.gravity = p.gravity@self.modes

    def residual(self, problem, displacement, force, parameter):
        """Virtual work on relative motions, integrating actual surface forces.

        Bulk stress does zero work on each constituent's rigid modes. This
        identity removes the expensive bulk evaluation, not a force equation.
        """
        p = self.physics
        nodal = displacement.reshape(*displacement.shape[:-1], -1, 3)
        jump = torch.einsum("qa,bsfac->bsfqc", p.shape, nodal[:, :, p.upper]-nodal[:, :, p.lower])
        material = problem.base.materials(parameter)
        traction = p.interface_path(jump.flatten(2, 3), material[:, 1, None, None])[0].reshape(jump.shape)
        interface = torch.einsum("bsfqc,fqck,f,q->bsk", traction, self.jump_modes, p.area, p.surface_weights)
        wall = torch.einsum("bsfq,fqk,f,q->bsk", p.wall_gap(displacement).clamp_max(0),
            self.wall_modes, p.wall_area, p.surface_weights)*p.settings["geometry"]["cone_contact_stiffness_pa_per_m"]
        gap = torch.einsum("qa,bsfa->bsfq", p.shape, nodal[:, :, p.trestle_faces, 2])
        support = torch.einsum("bsfq,fqk,f,q->bsk", gap.clamp_max(0), self.support_modes,
            p.trestle_face_area, p.surface_weights)*p.settings["geometry"]["trestle_stiffness_pa_per_m"]
        guide = (displacement@self.guide.T)@self.guide_modes*p.settings["geometry"]["lateral_guide_stiffness_n_per_m"]
        return (interface+wall+support+guide+p.anchor_loads(displacement)@self.anchor_modes
                -self.gravity-force[..., None]*self.jack_modes)

    def fold_update(self, coefficients):
        """Return the original output-layer coefficient increment; reference is zero."""
        update = self.encoded@coefficients
        reference = update.new_zeros((1, update.shape[1], update.shape[2]))
        return torch.cat([reference, update], dim=0).flatten(0, 1)

    def force_projection(self, coordinates, forces, moments):
        """Project whitened residuals using the saved body force/moment sums."""
        centred = np.asarray(moments)-np.cross(self.centre, forces)
        resultants = np.concatenate([np.concatenate([forces[..., b, :], centred[..., b, :]], axis=-1)
                                     for b in (0, 2)], axis=-1)
        scale = coordinates.coordinate_scale*self.physics.scale
        return resultants@self.transform/scale**2


class MaterialBodyHistory:
    def __init__(self, problem, network, basis, parameter):
        self.problem, self.network, self.basis = problem, network, basis
        self.parameter = parameter[None]
        with torch.no_grad():
            self.state = network(self.parameter)[0].detach()
        self.point = self.state.new_zeros((self.state.shape[1]-1, 12), requires_grad=True)
        self.value = self.function(self.point)

    def field(self, coefficients):
        shift = coefficients@self.basis.modes.T
        displacement = self.state+torch.cat([torch.zeros_like(self.state[:, :1]), shift[None]], dim=1)
        return self.network.balance(displacement)

    def function(self, coefficients):
        displacement = self.field(coefficients)
        return self.basis.residual(self.problem, displacement, self.network.forces(displacement),
                                   self.parameter)[0, 1:]

    def matvec(self, direction):
        direction = self.point.new_tensor(np.asarray(direction).reshape(self.point.shape))
        return torch.autograd.functional.jvp(self.function, self.point, direction)[1].detach().numpy()

    def rmatvec(self, direction):
        direction = self.point.new_tensor(np.asarray(direction).reshape(self.value.shape))
        return torch.autograd.grad(self.value, self.point, direction, retain_graph=True)[0].detach().numpy()

    def dense_jacobian(self):
        """Small history block for diagnosing shared versus local motion solves."""
        return np.stack([self.rmatvec(row).ravel() for row in np.eye(self.value.numel())])


class SharedBodyLinearization(LinearOperator):
    def __init__(self, problem, network, parameters, basis):
        with torch.no_grad():
            self.features = (network.displacement.features(parameters)@network.displacement.feature_transform).numpy()
        self.coefficient_shape = (len(problem.initial)-1, 12, self.features.shape[-1])
        self.histories = [MaterialBodyHistory(problem, network, basis, eta) for eta in parameters]
        self.residual_shape = (len(parameters), *self.coefficient_shape[:2])
        self.normalization = np.sqrt(np.prod(self.residual_shape))
        self.residual = np.stack([row.value.detach().numpy() for row in self.histories]).ravel()/self.normalization
        self.products = dict(forward=0, transpose=0)
        super().__init__(dtype=np.dtype(float), shape=(self.residual.size, int(np.prod(self.coefficient_shape))))

    def _matvec(self, vector):
        coefficients = np.asarray(vector).reshape(self.coefficient_shape)
        directions = np.einsum("mf,skf->msk", self.features, coefficients)
        values = np.stack([row.matvec(direction) for row, direction in zip(self.histories, directions)])
        self.products["forward"] += 1
        if self.products["forward"] % 5 == 0:
            print(json.dumps(dict(body_history_products=self.products)), flush=True)
        return values.ravel()/self.normalization

    def _rmatvec(self, vector):
        values = np.asarray(vector).reshape(self.residual_shape)
        products = np.stack([row.rmatvec(value) for row, value in zip(self.histories, values)])
        self.products["transpose"] += 1
        return np.einsum("mf,msk->skf", self.features, products).ravel()/self.normalization

    @torch.no_grad()
    def objective(self, coefficients):
        directions = np.einsum("mf,skf->msk", self.features, coefficients)
        return float(np.mean([row.function(row.point.new_tensor(direction)).square().mean().item()
                              for row, direction in zip(self.histories, directions)]))


def body_history_direction(problem, network, parameters, basis, tolerance):
    linear = SharedBodyLinearization(problem, network, parameters, basis)
    result = lsmr(linear, -linear.residual, atol=tolerance, btol=tolerance)
    update, stop, iterations, residual_norm, normal_norm, operator_norm, condition, norm = result
    image = linear@update
    details = dict(method="lsmr", tolerance=tolerance, stop_code=int(stop), iterations=int(iterations),
        products=linear.products.copy(), coefficient_shape=linear.coefficient_shape,
        residual_shape=linear.residual_shape,
        objective_before=float(linear.residual@linear.residual),
        linearized_objective_after=float(np.sum((linear.residual+image)**2)),
        directional_derivative=float(2*linear.residual@image), residual_norm=float(residual_norm),
        normal_residual_norm=float(normal_norm), operator_norm_estimate=float(operator_norm),
        condition_estimate=float(condition), update_norm=float(norm))
    return update.reshape(linear.coefficient_shape), linear, details
