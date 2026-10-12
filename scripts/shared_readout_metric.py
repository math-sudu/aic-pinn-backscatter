"""Frozen response residual metrics for the installed shunt circuit.

Apply the augmented inverse and its transpose to loss vectors. This is the
same quadratic form as storing every output adjoint, without dense adjoint
arrays. Only nonlinear sparse blocks vary across training conditions; the
elastic bulk matrix and preconditioner are shared.
"""

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gcrotmk, splu
import torch

from functional_readout import PathLinearization


def block_product(blocks, value, transpose=False):
    count = len(blocks)
    if transpose:
        return np.stack([sum(blocks[k][n].T@value[k] for k in range(n, count))
                         for n in range(count)])
    return np.stack([sum(blocks[n][m]@value[m] for m in range(n+1))
                     for n in range(count)])


class SparseMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, matrix):
        ctx.matrix = matrix
        return value.new_tensor((matrix@value.detach().cpu().numpy().T).T)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.new_tensor((ctx.matrix.T@gradient.detach().cpu().numpy().T).T), None


class PathInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, inverse):
        ctx.inverse = inverse
        return value.new_tensor(inverse.solve(value.detach().cpu().numpy()))

    @staticmethod
    def backward(ctx, gradient):
        return gradient.new_tensor(ctx.inverse.solve(gradient.detach().cpu().numpy(),
                                                     transpose=True)), None


class CouplingProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, blocks):
        ctx.blocks = blocks
        return value.new_tensor(block_product(blocks, value.detach().cpu().numpy()))

    @staticmethod
    def backward(ctx, gradient):
        return gradient.new_tensor(block_product(ctx.blocks, gradient.detach().cpu().numpy(),
                                                 transpose=True)), None


class FrozenHistoryInverse:
    def __init__(self, linear, bulk, coordinates, rtol, preconditioners=None):
        self.blocks = linear.blocks
        self.bulk, self.coordinates, self.rtol = bulk, coordinates, rtol
        self.preconditioners = preconditioners
        if preconditioners is not None and not preconditioners:
            # The nominal history decouples in reflection sectors. Factoring
            # these blocks separately avoids fill from roundoff cross-sector
            # entries. Other conditions still use their full coupled Jacobian.
            boundaries = np.r_[0, np.cumsum(coordinates.sizes)]
            preconditioners.extend([splu(row[n][lo:hi, lo:hi].tocsc(),
                permc_spec="MMD_AT_PLUS_A") for lo, hi in zip(boundaries[:-1], boundaries[1:])]
                for n, row in enumerate(self.blocks))
        # Avoid a separate copy of the large, invariant volume matrix per step
        # and condition. Earlier-time blocks contain only cohesive history.
        for n in range(len(self.blocks)):
            self.blocks[n][n] = (self.blocks[n][n]-bulk).tocsc()
            for block in self.blocks[n][:n+1]:
                block.eliminate_zeros()
        self.calls = dict(forward=0, transpose=0)
        self.operator_products = dict(forward=0, transpose=0)
        self.seconds = 0.

    def precondition(self, value, factors=None, transpose=False):
        parts = np.split(value, np.cumsum(self.coordinates.sizes)[:-1])
        if factors is not None:
            return np.concatenate([factor.solve(part, trans="T" if transpose else "N")
                for factor, part in zip(factors, parts)])
        return np.concatenate([factor.decode(factor.whiten(part))
            for factor, part in zip(self.coordinates.factors, parts)])

    def solve(self, rhs, transpose=False):
        started = time.perf_counter()
        role = "transpose" if transpose else "forward"
        count, size = rhs.shape
        result = np.zeros_like(rhs)
        metric = LinearOperator((size, size), matvec=self.precondition, dtype=float)
        bulk = self.bulk.T if transpose else self.bulk
        for n in (reversed(range(count)) if transpose else range(count)):
            value = rhs[n].copy()
            if transpose:
                for k in range(n+1, count):
                    value -= self.blocks[k][n].T@result[k]
            else:
                for m in range(n):
                    value -= self.blocks[n][m]@result[m]
            block = self.blocks[n][n].T if transpose else self.blocks[n][n]
            def product(x):
                self.operator_products[role] += 1
                return bulk@x+block@x

            operator = LinearOperator((size, size), matvec=product, dtype=float)
            if self.preconditioners is not None:
                metric = LinearOperator((size, size), dtype=float,
                    matvec=lambda x: self.precondition(x, self.preconditioners[n], transpose))
            # Share the nominal factors, but start each right-hand side with
            # an empty Krylov space. Cross-right-hand-side recycling diverged
            # in training even after rebuilding the stored operator images.
            result[n], status = gcrotmk(operator, value, M=metric, rtol=self.rtol,
                atol=0.)
            if status:
                raise RuntimeError(f"Readout {role} solve failed at history step {n}: {status}")
        self.calls[role] += 1
        self.seconds += time.perf_counter()-started
        return result


def circuit_derivatives(readout, raw, raw_tangents):
    """Instantaneous F_e and F_ee e_eta; steps do not share circuit states."""
    point = raw.detach().requires_grad_(True)

    def jacobian(value):
        outputs = readout(value).flatten(1)
        return torch.stack([torch.autograd.grad(outputs[:, k].sum(), value,
            retain_graph=True, create_graph=True)[0] for k in range(outputs.shape[1])], dim=1)

    gradient = jacobian(point).detach()
    curvature = [torch.autograd.functional.jvp(jacobian, point, direction)[1].detach()
                 for direction in raw_tangents]
    return gradient, curvature


class CaseReadoutMetric:
    def __init__(self, problem, readout, eta, displacement, velocities, bulk, rtol,
                 preconditioners=None):
        p, coordinates = problem.physics, problem.coordinates
        self.projection = coordinates.projection
        scale = problem.bond_scale(eta)
        linear = PathLinearization(p, displacement, bond_scale=scale,
            projection=self.projection, factorize=False)
        self.inverse = FrozenHistoryInverse(linear, bulk, coordinates, rtol, preconditioners)
        self.couplings = []
        for axis, velocity in enumerate(velocities):
            direction = torch.zeros_like(eta)
            direction[:, axis] = 1
            _, bond_tangent = torch.autograd.functional.jvp(problem.bond_scale, eta, direction)
            coupling = PathLinearization(p, displacement, bond_scale=scale,
                projection=self.projection, factorize=False, direction=(velocity, bond_tangent))
            for n, row in enumerate(coupling.blocks):
                for block in row[:n+1]:
                    block.eliminate_zeros()
            self.couplings.append(coupling.blocks)
        self.q_map = (self.projection.T@problem.functionals.q_matrix[:, p.free].detach().cpu().numpy().T).T
        self.raw_map = (self.projection.T@problem.raw_matrix[:, p.free].detach().cpu().numpy().T).T
        raw = displacement@problem.raw_matrix.T
        raw_tangents = [velocity@problem.raw_matrix.T for velocity in velocities]
        self.gradient, self.curvature = circuit_derivatives(readout, raw, raw_tangents)
        self.q_map = p.tensor(self.q_map)
        self.raw_map = p.tensor(self.raw_map)

    def output(self, displacement):
        host = displacement@self.q_map.T
        circuit = torch.einsum("sog,sg->so", self.gradient, displacement@self.raw_map.T)
        return torch.cat([host, circuit], dim=-1)

    def output_curvature(self, displacement, axis):
        circuit = torch.einsum("sog,sg->so", self.curvature[axis], displacement@self.raw_map.T)
        return torch.cat([torch.zeros_like(circuit), circuit], dim=-1)

    def state_correction(self, force):
        reduced = SparseMap.apply(force, self.projection.T)
        return -PathInverse.apply(reduced, self.inverse)

    def tangent_correction(self, delta, tangent, axis):
        reduced = SparseMap.apply(tangent, self.projection.T)
        rhs = reduced+CouplingProduct.apply(delta, self.couplings[axis])
        velocity_delta = -PathInverse.apply(rhs, self.inverse)
        return self.output(velocity_delta)+self.output_curvature(delta, axis)


class SharedReadoutMetric:
    def __init__(self, problem, readout, network, parameters, weights, rtol=1e-6):
        p, coordinates = problem.physics, problem.coordinates
        projection = coordinates.projection
        free = p.free.cpu().numpy()
        bulk = (projection.T@p._bulk_matrix[free][:, free]@projection).tocsc()
        self.cases, self.coefficient = [], None
        self.preconditioners = []
        regions = p.settings["observations"]["regions_m"]
        self.response_scale = p.scale/(max(r[1] for r in regions)-min(r[0] for r in regions))
        self.initial = dict(physics=0., sensitivity=[0., 0., 0.],
                            readout_value=0., readout_tangent=[0., 0., 0.])
        started = time.perf_counter()
        for i, eta in enumerate(parameters.split(1)):
            velocities, forces = [], []
            for axis in range(3):
                direction = torch.zeros_like(eta)
                direction[:, axis] = 1
                displacement, velocity = torch.autograd.functional.jvp(
                    lambda value: problem.displacement(network, value), eta, direction)
                force, tangent = problem.force_tangent(network, eta, axis)
                velocities.append(velocity[0])
                forces.append(tangent[0])
            print(f"readout metric {i+1}: state tangents ready", flush=True)
            case = CaseReadoutMetric(problem, readout, eta, displacement[0], velocities, bulk,
                                    rtol, self.preconditioners)
            print(f"readout metric {i+1}: linearization ready", flush=True)
            self.cases.append(case)
            with torch.no_grad():
                delta = case.state_correction(force[0])
                self.initial["physics"] += float(weights[i]*1e5*
                    coordinates.whiten(force).square().mean())
                self.initial["readout_value"] += float(weights[i]*self.goal(case.output(delta)))
                for axis, tangent in enumerate(forces):
                    self.initial["sensitivity"][axis] += float(weights[i]*1e5*
                        coordinates.whiten(tangent[None]).square().mean())
                    self.initial["readout_tangent"][axis] += float(weights[i]*self.goal(
                        case.tangent_correction(delta, tangent, axis)))
            print(f"readout metric {i+1}/{len(parameters)}: {time.perf_counter()-started:.1f} s", flush=True)
        self.refresh_seconds = time.perf_counter()-started
        self.coefficient = ((self.initial["physics"]+sum(self.initial["sensitivity"])) /
            (self.initial["readout_value"]+sum(self.initial["readout_tangent"])))

    def goal(self, correction):
        return (correction/self.response_scale).square().mean()*1e5

    def accumulate_loss(self, problem, network, parameters, weights, backward=True):
        totals = dict(physics=0., sensitivity=[0., 0., 0.],
                      readout_value=0., readout_tangent=[0., 0., 0.])
        started, last_log = time.perf_counter(), time.perf_counter()
        for i, (eta, case) in enumerate(zip(parameters.split(1), self.cases)):
            tangents = []
            for axis in range(3):
                current_force, tangent = problem.force_tangent(network, eta, axis,
                                                               create_graph=backward)
                if axis == 0:
                    force = current_force
                tangents.append(tangent)
            # Share the state inverse and its transpose across all three
            # tangent terms. Their force gradients add before that transpose.
            delta = case.state_correction(force[0])
            physical = weights[i]*1e5*problem.coordinates.whiten(force).square().mean()
            goal_value = weights[i]*self.goal(case.output(delta))
            totals["physics"] += float(physical.detach())
            totals["readout_value"] += float(goal_value.detach())
            loss = physical+self.coefficient*goal_value
            for axis, tangent in enumerate(tangents):
                derivative = problem.coordinates.whiten(tangent)
                sensitivity = weights[i]*1e5*derivative.square().mean()
                goal_tangent = weights[i]*self.goal(case.tangent_correction(delta, tangent[0], axis))
                totals["sensitivity"][axis] += float(sensitivity.detach())
                totals["readout_tangent"][axis] += float(goal_tangent.detach())
                loss = loss+sensitivity+self.coefficient*goal_tangent
            if backward:
                loss.backward()
            if time.perf_counter()-last_log >= 60:
                print(f"readout objective {i+1}/{len(parameters)}: "
                      f"{time.perf_counter()-started:.1f} s", flush=True)
                last_log = time.perf_counter()
        problem.last_loss_terms = totals
        return parameters.new_tensor(totals["physics"]+sum(totals["sensitivity"])+
            self.coefficient*(totals["readout_value"]+sum(totals["readout_tangent"])))

    def work(self):
        return dict(refresh_seconds=self.refresh_seconds, coefficient=self.coefficient,
            initial_terms=self.initial,
            path_solves={key: sum(case.inverse.calls[key] for case in self.cases)
                         for key in ("forward", "transpose")},
            inverse_solver="GCROT(m,k) with a fresh space per right-hand side and shared nominal-history sector LU",
            preconditioner_factor_entries=sum(factor.L.nnz+factor.U.nnz
                for step in self.preconditioners for factor in step),
            operator_products={key: sum(case.inverse.operator_products[key] for case in self.cases)
                               for key in ("forward", "transpose")},
            solve_seconds=sum(case.inverse.seconds for case in self.cases))


class SharedDesignReadoutMetric(SharedReadoutMetric):
    """Preserve host and circuit-minus-host values and finite changes.

    The installed circuit and history Jacobians stay fixed within a segment.
    Only residual corrections are penalized; predicted circuit mismatch is
    never a neural training target. Each output block is balanced against the
    global physics term once, at the initial state.
    """
    def __init__(self, problem, readout, network, parameters, weights, rtol=1e-6):
        if len(parameters) < 2 or torch.any(parameters[0] != 0):
            raise ValueError("Finite-response training needs the nominal condition first and other conditions")
        p, coordinates = problem.physics, problem.coordinates
        projection = coordinates.projection
        free = p.free.cpu().numpy()
        bulk = (projection.T@p._bulk_matrix[free][:, free]@projection).tocsc()
        regions = p.settings["observations"]["regions_m"]
        self.response_scale = p.scale/(max(r[1] for r in regions)-min(r[0] for r in regions))
        self.cases, self.preconditioners = [], []
        self.initial = self.empty_terms()
        started = time.perf_counter()
        nominal = None
        for i, eta in enumerate(parameters.split(1)):
            with torch.no_grad():
                displacement = problem.displacement(network, eta)[0]
                force = problem.force(network, eta)
            case = CaseReadoutMetric(problem, readout, eta, displacement, [], bulk,
                                    rtol, self.preconditioners)
            self.cases.append(case)
            with torch.no_grad():
                correction = self.correction(case, force)
                if i == 0:
                    nominal = correction
                terms = self.response_terms(correction, nominal, weights[i],
                    weights[i]/weights[1:].sum() if i else 0.)
                terms["physics"] = weights[i]*1e5*coordinates.whiten(force).square().mean()
                for key, value in terms.items():
                    self.initial[key] += float(value)
            print(f"design readout metric {i+1}/{len(parameters)}: "
                  f"{time.perf_counter()-started:.1f} s", flush=True)
        self.refresh_seconds = time.perf_counter()-started
        self.coefficient = {block: self.initial["physics"]/(
            self.initial[f"{block}_value"]+self.initial[f"{block}_finite_change"])
            for block in ("host", "discrepancy")}

    @staticmethod
    def empty_terms():
        return dict(physics=0., host_value=0., host_finite_change=0.,
                    discrepancy_value=0., discrepancy_finite_change=0.)

    @staticmethod
    def correction(case, force):
        host, circuit = case.output(case.state_correction(force[0])).chunk(2, dim=-1)
        return torch.stack([host, circuit-host], dim=-1)

    def response_terms(self, correction, nominal, weight, change_weight):
        terms = {}
        for axis, block in enumerate(("host", "discrepancy")):
            terms[f"{block}_value"] = weight*self.goal(correction[..., axis])
            terms[f"{block}_finite_change"] = change_weight*self.goal(
                correction[..., axis]-nominal[..., axis])
        return terms

    def objective(self, terms):
        return terms["physics"]+sum(self.coefficient[block]*(
            terms[f"{block}_value"]+terms[f"{block}_finite_change"])
            for block in ("host", "discrepancy"))

    def accumulate_loss(self, problem, network, parameters, weights, backward=True):
        totals = self.empty_terms()
        started, last_log = time.perf_counter(), time.perf_counter()
        with torch.set_grad_enabled(backward):
            nominal_force = problem.force(network, parameters[:1])
            nominal = self.correction(self.cases[0], nominal_force)
            # Accumulate the shared nominal cotangent on this small leaf.
            # Its single history transpose is applied after all other cases,
            # so only two physical residual graphs are ever retained together.
            reference = nominal.detach().requires_grad_(backward)
            for i in range(1, len(self.cases)):
                force = problem.force(network, parameters[i:i+1])
                correction = self.correction(self.cases[i], force)
                terms = self.response_terms(correction, reference, weights[i],
                                             weights[i]/weights[1:].sum())
                terms["physics"] = weights[i]*1e5*problem.coordinates.whiten(force).square().mean()
                for key, value in terms.items():
                    totals[key] += float(value.detach())
                if backward:
                    self.objective(terms).backward()
                if time.perf_counter()-last_log >= 60:
                    print(f"design readout objective {i+1}/{len(parameters)}: "
                          f"{time.perf_counter()-started:.1f} s", flush=True)
                    last_log = time.perf_counter()
            terms = self.response_terms(nominal, nominal, weights[0], 0.)
            terms["physics"] = weights[0]*1e5*problem.coordinates.whiten(nominal_force).square().mean()
            for key, value in terms.items():
                totals[key] += float(value.detach())
            if backward:
                torch.autograd.backward((self.objective(terms), nominal),
                                        (None, reference.grad))
        problem.last_loss_terms = totals
        return parameters.new_tensor(self.objective(totals))
