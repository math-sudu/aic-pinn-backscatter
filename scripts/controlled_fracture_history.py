"""Shared displacement/load network with reactive apparatus and volume fracture."""

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
import torch
from torch import nn

from fixed_reference_history import FixedReferenceNetwork, MaterialHistoryProblem, ROOT, absolute_components, sparse_tensor
from fracture_path import ResidualTensionLaw, TensileCapLaw, displacement_constraint
from spatial_fracture import BodyFracture
from rigid_reaction_balance import RigidReactionBalance
from constituent_reaction_balance import ConstituentReactionBalance


class ControlledHistoryNetwork(nn.Module):
    """Reuse every displacement coefficient and add a shared trainable load head."""

    def __init__(self, displacement_network, initial_force, load_scale_n):
        super().__init__()
        self.displacement = displacement_network
        p = displacement_network.coordinates.physics
        self.load_scale = load_scale_n
        self.register_buffer("reference_force", p.tensor(initial_force[0]))
        self.load_layer = nn.Linear(displacement_network.layers[-1].in_features, len(initial_force)-1)
        self.load_layer.to(dtype=p.dtype, device=p.device)
        with torch.no_grad():
            self.load_layer.weight.zero_()
            self.load_layer.bias.copy_(p.tensor(initial_force[1:])/self.load_scale)

    def forward(self, parameters):
        displacement = self.displacement(parameters)
        features = self.displacement.features(parameters)
        coefficient = torch.cat([self.load_layer.weight, self.load_layer.bias[:, None]], dim=-1)
        load = (features@coefficient.T)*self.load_scale
        return displacement, torch.cat([self.reference_force.expand(len(parameters), 1), load], dim=1)


class ExactControlNetwork(nn.Module):
    """Eliminate the series-compliance equation without displacement labels.

    The reference is either learned jointly or retained as a prepared material
    function. Its derivative enters every later force through the same
    work-conjugate jack operator.
    """

    def __init__(self, displacement_network, problem):
        super().__init__()
        self.displacement = displacement_network
        self.compliance = problem.config["apparatus"]["compliance_m_per_n"]
        if self.compliance <= 0:
            raise ValueError("Exact load elimination requires positive series compliance")
        self.register_buffer("jack_operator", problem.physics.jack_operator.clone())
        self.register_buffer("machine", problem.machine.clone())
        self.register_buffer("reference_force", problem.physics.tensor(problem.reference_force))
        self.register_buffer("reference_coefficients", None)
        self.rigid_balance = (RigidReactionBalance(problem) if problem.config.get("rigid_balance") else None)
        self.constituent_balance = (ConstituentReactionBalance(problem)
                                    if problem.config.get("constituent_balance") else None)
        self.materials = problem.base.materials

    @torch.no_grad()
    def fix_reference(self):
        self.displacement.layers[:-1].requires_grad_(False)
        last = self.displacement.layers[-1]
        width = self.displacement.shape[-1]
        self.reference_coefficients = torch.cat([last.weight[:width], last.bias[:width, None]], dim=-1).clone()

    def raw(self, parameters):
        raw = self.displacement.raw(parameters)
        if self.reference_coefficients is not None:
            features = self.displacement.features(parameters)@self.displacement.feature_transform
            reference = features@self.reference_coefficients.T
            raw = torch.cat([reference[:, None], raw[:, 1:]], dim=1)
        return raw

    def forces(self, displacement):
        increment = (displacement[:, 1:]-displacement[:, :1])@self.jack_operator
        force = self.reference_force+(self.machine[None, 1:]-increment)/self.compliance
        return torch.cat([self.reference_force.expand(len(displacement), 1), force], dim=1)

    def forward(self, parameters):
        raw = self.raw(parameters)
        if self.displacement.path_coordinates == "reference_increments":
            raw = absolute_components(raw)
        displacement = self.displacement.coordinates.decode(raw)*self.displacement.coordinates.physics.scale
        displacement = self.balance(displacement, parameters)
        return displacement, self.forces(displacement)

    def balance(self, displacement, parameters=None):
        if self.constituent_balance is not None:
            if parameters is None:
                raise ValueError("Constituent equilibrium requires the material input")
            return self.constituent_balance(displacement, self.materials(parameters)[:, 1])
        return self.rigid_balance(displacement) if self.rigid_balance is not None else displacement


class ControlledFractureProblem:
    def __init__(self, config):
        self.config = config
        self.source = json.loads((ROOT/config["source_network"]).read_text())
        # Keep this invertible basis so old checkpoints retain their function.
        # Its rigid gauge is a coordinate metric only, never a physical force.
        self.base = MaterialHistoryProblem(self.source["setup"])
        self.physics = p = self.base.physics
        guides = config.get("lateral_guides", "independent_lower_bodies")
        if guides == "substrate_only":
            # Both mesh builders order the three substrate rows before the
            # three cone rows. Remove the cone's physical restraint after
            # preparing the unchanged neural coordinate metric.
            operator = p.mesh["lateral_guide_operator"]
            removed = sparse.csr_matrix(operator[3:])
            stiffness = p.settings["geometry"]["lateral_guide_stiffness_n_per_m"]
            correction = stiffness*(removed.T@removed)
            p.mesh = dict(p.mesh, lateral_guide_operator=operator[:3].copy())
            p._bulk_matrix = (p._bulk_matrix-correction).tocsr()
            p.intact_bulk_interface = (p.intact_bulk_interface-correction).tocsr()
            p.stiffness = sparse_tensor(p._bulk_matrix, p)
            p.diagonal = p.tensor(p._bulk_matrix.diagonal())
        elif guides != "independent_lower_bodies":
            raise ValueError(f"Unknown lateral guide configuration: {guides}")
        p.interface_friction_coefficient = config.get("interface_friction_coefficient", 0.)
        if p.interface_friction_coefficient < 0:
            raise ValueError("Interface friction coefficient must be nonnegative")
        if "far_field_shear_release_radius_m" in config:
            radius = config["far_field_shear_release_radius_m"]
            points = np.einsum("qa,fac->fqc", p.shape.numpy(),
                               p.mesh["xyz"][p.mesh["interface_upper"]]).reshape(-1, 3)
            released = np.linalg.norm(points[:, :2], axis=-1) > radius
            if not np.isfinite(radius) or radius <= 0 or not released.any():
                raise ValueError("Shear release needs a positive radius inside the interface")
            scale = np.ones_like(points)
            scale[released, :2] = 0.
            p.interface_coordinate_scale = p.tensor(scale)
        if "anchor_reference_m" in config:
            p.anchor_reference = p.tensor(config["anchor_reference_m"])
        self.coordinates = self.base.coordinates
        self.initial = self.base.initial
        self.boundary = self.base.boundary
        material = json.loads((ROOT/self.source["setup"]["material_configuration"]).read_text())
        boundary = json.loads((ROOT/material["boundary_configuration"]).read_text())
        asymmetric = json.loads((ROOT/boundary["asymmetric_configuration"]).read_text())
        support = json.loads((ROOT/asymmetric["support_report"]).read_text())
        selected = next(row for row in support["cases"] if
                        row["rod_stiffness_n_per_m"] == asymmetric["conditional_rod_stiffness_n_per_m"])
        # ForceHistoryProblem had disabled the physical trestles. Restore them,
        # and retain InitialStatePhysics's seated reactive rod springs.
        p.settings["geometry"]["trestle_stiffness_pa_per_m"] = selected["trestle_stiffness_pa_per_m"]
        material_config = json.loads((ROOT/config["fracture_configuration"]).read_text())
        scenario = material_config["published_tension_scenario"]
        law = ResidualTensionLaw(**{key: scenario[key] for key in
            ("tensile_strength_pa", "a1_per_m", "a2_per_m", "b2")})
        self.fracture = BodyFracture(p, law, config["nonlocal_radius_m"], config["softening_length_m"])
        self.substrate_fracture = None
        if "substrate_tension_cap_pa" in config:
            self.substrate_fracture = BodyFracture(p, TensileCapLaw(config["substrate_tension_cap_pa"]),
                config["nonlocal_radius_m"], config["softening_length_m"], body=1)
            self.substrate_stiffness_matrix = self.substrate_fracture.elastic_matrix(
                p.settings["bulk"]["substrate_young_pa"])
            self.substrate_stiffness = sparse_tensor(self.substrate_stiffness_matrix, p)
        self.machine = p.tensor(self.boundary["measured_jack_increment_mm"]*1e-3)
        self.reference_force = float(self.boundary["force_n"][0])

    def load_network(self, *, exact_control=False, fixed_reference=False):
        setup = self.source["setup"]
        network = FixedReferenceNetwork(self.coordinates, self.initial, setup["width"], setup["path_coordinates"])
        path = (ROOT/self.config["source_network"]).with_suffix(".pt")
        network.load_state_dict(torch.load(path, weights_only=True)["network"])
        if exact_control:
            controlled = ExactControlNetwork(network, self)
            if fixed_reference:
                controlled.fix_reference()
            return controlled
        return ControlledHistoryNetwork(network, self.boundary["force_n"], self.config["load_scale_n"])

    def equilibrium(self, displacement, force, parameters):
        p = self.physics
        material = self.base.materials(parameters)
        internal, interface_damage, interface_history = p.forces(displacement,
            compression_stiffness=material[:, 1, None, None])
        elastic_top = torch.sparse.mm(self.base.top_stiffness,
            displacement.reshape(-1, p.ndof).T).T.reshape(displacement.shape)
        top, bulk_state = self.fracture.force(displacement, material[:, 0])
        anchors = p.anchor_loads(displacement)@p.anchor_operator
        residual = internal-elastic_top+top+anchors-p.gravity-force[..., None]*p.jack_operator
        substrate_state = {}
        if self.substrate_fracture is not None:
            correction, substrate_state = self.substrate_correction(displacement)
            residual = residual+correction
        return residual, dict(interface_damage=interface_damage, interface_history=interface_history,
                             **{"bulk_"+name: value for name, value in bulk_state.items()},
                             **{"substrate_"+name: value for name, value in substrate_state.items()})

    def substrate_correction(self, displacement):
        nonlinear, state = self.substrate_fracture.force(displacement,
            self.physics.tensor(self.physics.settings["bulk"]["substrate_young_pa"]))
        elastic = torch.sparse.mm(self.substrate_stiffness,
            displacement.reshape(-1, self.physics.ndof).T).T.reshape(displacement.shape)
        return nonlinear-elastic, state

    def control(self, displacement, force):
        return displacement_constraint(displacement[:, 1:], force[:, 1:], self.physics.jack_operator,
            self.machine[None, 1:], reference_displacement=displacement[:, :1],
            reference_load_n=self.reference_force,
            compliance_m_per_n=self.config["apparatus"]["compliance_m_per_n"])

    def residual(self, network, parameters):
        displacement, force = network(parameters)
        residual, _ = self.equilibrium(displacement, force, parameters)
        return (self.coordinates.whiten(residual), self.control(displacement, force)/self.config["control_scale_m"])

    def pack(self, displacement, force):
        """Physical unknowns; the first force is known at the loaded reference."""
        return torch.cat([displacement.ravel(), force[1:]])

    def unpack(self, values):
        nstep, ndof = self.initial.shape
        displacement = values[:nstep*ndof].reshape(nstep, ndof)
        force = torch.cat([values.new_tensor([self.reference_force]), values[nstep*ndof:]])
        return displacement, force

    def physical_residual(self, values, parameters):
        displacement, force = self.unpack(values)
        residual = self.equilibrium(displacement[None], force[None], parameters[None])[0]
        return torch.cat([residual.ravel(), self.control(displacement[None], force[None]).ravel()])

    def linearization(self, displacement, force, parameters):
        """Matrix-free full Jacobian and transpose, including nonlocal/time links.

        A local block PathLinearization would omit the spatial averaging and
        apparatus terms. This operator differentiates the actual residual.
        Its mechanical rows have N units and control rows have m units.
        """
        point = self.pack(displacement, force).detach().requires_grad_(True)
        eta = self.physics.tensor(parameters).detach()
        function = lambda value: self.physical_residual(value, eta)
        value = function(point)

        def multiply(direction):
            return torch.autograd.functional.jvp(function, point,
                self.physics.tensor(np.asarray(direction).reshape(-1)))[1].detach().numpy()

        def transpose(direction):
            return torch.autograd.grad(value, point,
                self.physics.tensor(np.asarray(direction).reshape(-1)), retain_graph=True)[0].detach().numpy()

        return LinearOperator((len(point), len(point)), matvec=multiply, rmatvec=transpose, dtype=float)
