"""Causal control-conditioned fields with local displacement and reactive loads.

A shared recurrent encoder reads physical controls, never a step or branch ID.
The spatial head is common to every subsequent state. A supplied equilibrium
prefix fixes the initial condition; the full field history enters the original
bulk and interface laws. Local control replaces the machine-control equation.
"""

import numpy as np
import torch
from torch import nn

from constituent_reaction_balance import ConstituentReactionBalance


class LocalBalanceError(RuntimeError):
    """The constituent/local-control equations did not close for a trial."""


class LocalControlBalance:
    """Implicit body motions and jack load at prescribed local displacement."""

    def __init__(self, problem):
        self.base = ConstituentReactionBalance(problem)
        self.local = problem.physics.relative_operator.mean(dim=0)
        self.local_modes = self.local@self.base.modes
        self.root_compliance = np.sqrt(self.base.compliance)

    def equations(self, unknown, offsets, prior, compression, local_offset, *, tangent=False):
        b, root = self.base, self.root_compliance
        motion, load = unknown[..., :18], unknown[..., 18]/root
        jack = offsets[..., -1]+motion@b.jack_modes
        # Setting target to the actual jack removes the apparatus stiffness.
        # Load is now independent; the apparatus coordinate is recovered later.
        result = b.equations(motion, offsets, prior, compression, jack, tangent=tangent)
        residual = result[0]-load[..., None]*b.jack_modes
        control = (local_offset+motion@self.local_modes)/root
        residual = torch.cat([residual, control[..., None]], dim=-1)
        if not tangent:
            return residual, result[1]
        matrix = unknown.new_zeros((*unknown.shape[:-1], 19, 19))
        matrix[..., :18, :18] = result[2]-torch.outer(b.jack_modes, b.jack_modes)/b.compliance
        matrix[..., :18, 18] = -b.jack_modes/root
        matrix[..., 18, :18] = self.local_modes/root
        return residual, result[1], matrix

    @torch.no_grad()
    def solve(self, offsets, prior, compression, local_offset):
        unknown = offsets.new_zeros((*offsets.shape[:-1], 19))
        for _ in range(80):
            residual, history, matrix = self.equations(
                unknown, offsets, prior, compression, local_offset, tangent=True)
            if float(residual.abs().max()) < 1e-10:
                return unknown, history, matrix
            direction = torch.linalg.solve(matrix, -residual[..., None])[..., 0]
            objective = residual.square().sum(-1)
            alpha = torch.ones_like(objective)
            for _ in range(50):
                trial = unknown+alpha[..., None]*direction
                value, _ = self.equations(trial, offsets, prior, compression, local_offset)
                accepted = ((value.square().sum(-1) <= (1-1e-4*alpha)*objective)
                            | (value.abs().amax(-1) < 1e-10))
                if bool(accepted.all()):
                    unknown = trial
                    break
                alpha = torch.where(accepted, alpha, .5*alpha)
            else:
                raise LocalBalanceError("Local constituent equilibrium line search failed")
        raise LocalBalanceError("Local constituent equilibrium did not converge")

    def __call__(self, fields, targets, prefix, compression):
        b = self.base
        prefix_offsets = b.boundary_values(prefix)
        jumps = prefix_offsets[..., :b.sizes[0]].reshape(*prefix.shape[:2], -1, 3)
        history = b.physics.interface_path(jumps, compression[:, None, None])[2][:, -1]
        offsets = b.boundary_values(fields)
        local_offset = (fields-prefix[:, :1])@self.local-targets
        displacements, forces = [], []
        for index in range(fields.shape[1]):
            unknown, history = LocalControlStep.apply(offsets[:, index], history, compression,
                                                      local_offset[:, index], self)
            displacements.append(fields[:, index]+unknown[:, :18]@b.modes.T)
            forces.append(unknown[:, 18]/self.root_compliance)
        return torch.stack(displacements, dim=1), torch.stack(forces, dim=1)


class LocalControlStep(torch.autograd.Function):
    """First implicit derivative, including the causal interface-history path."""

    @staticmethod
    def forward(ctx, offsets, prior, compression, local_offset, balance):
        unknown, history, matrix = balance.solve(offsets, prior, compression, local_offset)
        ctx.balance = balance
        ctx.save_for_backward(offsets, prior, compression, local_offset, unknown, matrix)
        return unknown, history

    @staticmethod
    def backward(ctx, unknown_gradient, history_gradient):
        create_graph = torch.is_grad_enabled()
        offsets, prior, compression, local_offset, unknown, matrix = ctx.saved_tensors
        with torch.enable_grad():
            inputs = tuple(value.detach().requires_grad_(True)
                           for value in (offsets, prior, compression, local_offset))
            point = unknown.detach().requires_grad_(True)
            residual, history = ctx.balance.equations(point, *inputs)
            history_pullback = torch.autograd.grad(history, point, history_gradient,
                retain_graph=True, create_graph=create_graph)[0]
            dual = torch.linalg.solve(matrix.transpose(-1, -2),
                (unknown_gradient+history_pullback)[..., None])[..., 0]
            gradients = torch.autograd.grad((residual, history), inputs, (-dual, history_gradient),
                                            create_graph=create_graph)
        return (*gradients, None)


class CausalControlEncoder(nn.Module):
    """Current control and signed increment drive a common recurrent state.

    Controls are displacements relative to the original loaded reference.
    The latent state represents the ordered discrete protocol; constitutive
    memory is separately evaluated from the entire spatial displacement path.
    """

    def __init__(self, width, scale_m=1e-3):
        super().__init__()
        self.scale_m = scale_m
        self.cell = nn.GRUCell(4, width)

    def forward(self, parameters, prefix_controls, controls):
        sequence = torch.cat([prefix_controls, controls], dim=1)/self.scale_m
        state = sequence.new_zeros((len(sequence), self.cell.hidden_size))
        previous = sequence[:, 0]
        outputs = []
        for index in range(sequence.shape[1]):
            current = sequence[:, index]
            inputs = torch.cat([parameters, current[:, None], (current-previous)[:, None]], dim=-1)
            state = self.cell(inputs, state)
            if index >= prefix_controls.shape[1]:
                outputs.append(state)
            previous = current
        return torch.stack(outputs, dim=1)


class ControlHistoryNetwork(nn.Module):
    """One spatial head, a supplied physical prefix, and variable-length controls.

    The displacement/hinge skip is initialized from three existing loading
    fields. It is only a warm start: subsequent optimization uses physical
    equations. No unloading field is an initialization or training label.
    """

    def __init__(self, problem, loading_fields, reference_field, width=8):
        super().__init__()
        self.problem, self.coordinates = problem, problem.coordinates
        p = problem.physics
        self.balance = LocalControlBalance(problem)
        self.encoder = CausalControlEncoder(width).to(dtype=p.dtype, device=p.device)
        self.head = nn.Linear(width+2, sum(self.coordinates.sizes)).to(dtype=p.dtype, device=p.device)
        control = ((loading_fields-reference_field)@self.balance.local).detach()
        self.register_buffer("origin_m", control[0])
        self.register_buffer("hinge_m", control[1])
        encoded = p.tensor(self.coordinates.encode_nominal(loading_fields.detach().cpu().numpy()))
        slope0 = (encoded[1]-encoded[0])/((control[1]-control[0])/self.encoder.scale_m)
        slope1 = (encoded[2]-encoded[1])/((control[2]-control[1])/self.encoder.scale_m)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[:, 0] = slope0
            self.head.weight[:, 1] = slope1-slope0
            self.head.bias.copy_(encoded[0])

    def features(self, parameters, prefix, controls):
        prefix_controls = (prefix-prefix[:, :1])@self.balance.local
        hidden = self.encoder(parameters, prefix_controls, controls)
        scale = self.encoder.scale_m
        direct = torch.stack([(controls-self.origin_m)/scale,
                              torch.relu(controls-self.hinge_m)/scale], dim=-1)
        return torch.cat([direct, hidden], dim=-1)

    def forward(self, parameters, prefix, prefix_forces, controls):
        raw = self.head(self.features(parameters, prefix, controls))
        return self.field_from_raw(raw, parameters, prefix, prefix_forces, controls)

    def field_from_raw(self, raw, parameters, prefix, prefix_forces, controls):
        fields = self.coordinates.decode(raw)*self.problem.physics.scale
        compression = self.problem.base.materials(parameters)[:, 1]
        fields, forces = self.balance(fields, controls, prefix, compression)
        return torch.cat([prefix, fields], dim=1), torch.cat([prefix_forces, forces], dim=1)

    def residual(self, parameters, prefix, prefix_forces, controls):
        fields, forces = self(parameters, prefix, prefix_forces, controls)
        residual = self.problem.equilibrium(fields, forces, parameters)[0][:, prefix.shape[1]:]
        return self.coordinates.whiten(residual)

    def machine_coordinates(self, fields, forces):
        p, config = self.problem.physics, self.problem.config
        return ((fields-fields[:, :1])@p.jack_operator
                +config["apparatus"]["compliance_m_per_n"]*(forces-forces[:, :1]))
