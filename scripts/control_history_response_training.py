"""Physics-generated regional targets for an actual control history.

One frozen Newton correction includes all earlier predicted-state derivatives.
It defines local H=(host, circuit-host) targets, never external field labels.
Value and equal-control history terms share both states' gradients.
"""

import json
import time

import numpy as np
import torch

from controlled_branch_solver import CurrentControlledTangent, IndirectControlTangent


def causal_local_correction(problem, fields, forces, parameters, prefix_states, tolerance=.02):
    """Solve the lower triangular physical path with zero local-control changes."""
    p = problem.physics
    compliance = problem.config["apparatus"]["compliance_m_per_n"]
    local, jack = p.relative_operator.mean(dim=0), p.jack_operator
    shift, load_shift = torch.zeros_like(fields), torch.zeros_like(forces)
    records = []
    for case in range(len(fields)):
        for step in range(prefix_states, fields.shape[1]):
            path, load = fields[case:case+1, :step+1], forces[case:case+1, :step+1]
            eta = parameters[case:case+1]

            def terminal(value):
                return problem.equilibrium(value, load, eta)[0][0, -1]

            residual, prior_image = torch.autograd.functional.jvp(
                terminal, path, shift[case:case+1, :step+1])
            tangent = CurrentControlledTangent(problem, path[0], eta[0])
            bordered = IndirectControlTangent(tangent, jack.numpy(), local.numpy(), compliance)
            direction, info = bordered.solve(np.r_[(residual+prior_image).detach().numpy(), 0.],
                                             tolerance, preconditioner="reference")
            shift[case, step] = p.tensor(direction[:-1])
            load_shift[case, step] = (direction[-1]-jack@shift[case, step])/compliance
            records.append(dict(case=case, step=step, **info))
            print(json.dumps(dict(control_history_correction=records[-1])), flush=True)
    return shift, load_shift, records


class ControlHistoryResponseMetric:
    def __init__(self, problem, readout, prefix_states, history_pairs, protocol_pairs=()):
        self.problem, self.readout = problem, readout
        self.prefix_states = prefix_states
        self.history_pairs = tuple(history_pairs)
        self.protocol_pairs = tuple(protocol_pairs)

    def values(self, fields):
        host = fields@self.problem.base.functionals.q_matrix.T
        circuit = self.readout(fields@self.problem.base.raw_matrix.T).flatten(2)
        host = host[:, self.prefix_states:]-host[:, :1]
        circuit = circuit[:, self.prefix_states:]-circuit[:, :1]
        return torch.stack([host, circuit-host], dim=-1)

    def changes(self, values):
        return torch.stack([values[:, later]-values[:, earlier]
                            for earlier, later in self.history_pairs], dim=1)

    def terms(self, values):
        return {f"{block}_{kind}": part[..., axis].square().mean()
                for axis, block in enumerate(("host", "discrepancy"))
                for kind, part in self.blocks(values)}

    def blocks(self, values):
        changes = self.changes(values)
        blocks = [("value", values), ("history_change", changes)]
        if self.protocol_pairs:
            protocol_changes = torch.stack([changes[later]-changes[earlier]
                                             for earlier, later in self.protocol_pairs])
            blocks.append(("protocol_change", protocol_changes))
        return blocks

    def pack(self, values):
        return torch.cat([part[..., axis].flatten()*np.sqrt(self.coefficients[block]/part[..., axis].numel())
                          for axis, block in enumerate(("host", "discrepancy"))
                          for _, part in self.blocks(values)])

    def objective(self, fields):
        return self.pack(self.values(fields)-self.target).square().sum()


def build_control_history_metric(network, parameters, prefix, prefix_forces, controls, readout,
                                 *, coefficients=None, tolerance=.02, history_pairs=((0, 2),),
                                 protocol_pairs=()):
    started = time.perf_counter()
    problem = network.problem
    for earlier, later in history_pairs:
        if not torch.allclose(controls[:, earlier], controls[:, later], rtol=0, atol=1e-12):
            raise ValueError("A history contrast must join the same local control value")
        for left, right in protocol_pairs:
            if not torch.allclose(controls[left, earlier], controls[right, earlier], rtol=0, atol=1e-12):
                raise ValueError("A protocol contrast must compare histories at the same local control")
    with torch.no_grad():
        fields, forces = network(parameters, prefix, prefix_forces, controls)
        residual = problem.equilibrium(fields, forces, parameters)[0][:, prefix.shape[1]:]
        physics_loss = float(problem.coordinates.whiten(residual).square().mean())
    correction, load_correction, records = causal_local_correction(
        problem, fields, forces, parameters, prefix.shape[1], tolerance)
    metric = ControlHistoryResponseMetric(problem, readout, prefix.shape[1], history_pairs, protocol_pairs)
    value, response_correction = torch.autograd.functional.jvp(metric.values, fields, correction)
    metric.target = (value+response_correction).detach()
    terms = {key: float(term) for key, term in metric.terms(response_correction).items()}
    metric.coefficients = coefficients or {
        block: physics_loss/sum(term for key, term in terms.items() if key.startswith(block+"_"))
        for block in ("host", "discrepancy")}
    metric.work = dict(history_pairs=[list(pair) for pair in history_pairs],
        protocol_pairs=[list(pair) for pair in protocol_pairs],
        coefficients=metric.coefficients, initial_terms=terms, physical_correction=records,
        local_correction_microstrain=(response_correction*1e6).tolist(),
        initial_response_microstrain=(value*1e6).tolist(),
        load_correction_kn=(load_correction[:, prefix.shape[1]:]/1e3).tolist(),
        refresh_seconds=time.perf_counter()-started,
        target_definition="H(U)+H'(U) delta_U, using the full causal physical Newton equation with fixed prefix and local controls; no external response labels. Targets refresh per accepted update; coefficients stay fixed.")
    return metric
