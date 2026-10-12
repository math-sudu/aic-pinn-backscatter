"""Loaded bridge functionals and the exact active-branch path adjoint.

The adjoint linearizes the neural physics equations at the current prediction.
It supplies local residual weights, not independent mechanical truth. The
lower-triangular blocks include derivatives of earlier interface maxima.
"""

import json

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import torch

from bridge_circuit import fixed_calibration
from stateful_physics import ROOT, cohesive_path


def bridge_voltage(strain, count, settings):
    """Torch equivalent of the previously KCL-verified loaded bridge."""
    s = settings
    rg = count*s["sensor_resistance_ohm"]
    rc = rg+s["lead_resistance_per_string_ohm"]
    rs = rg*(1+s["gauge_factor"]*strain)+s["lead_resistance_per_string_ohm"]
    return loaded_bridge_voltage(rs, rc, s)


def loaded_bridge_voltage(rs, rc, settings):
    """Differentiable loaded-node voltage for specified bridge resistances."""
    s = settings
    if torch.any(rs <= 0):
        raise ValueError("Non-positive sensor-string resistance")
    vs, vr = s["excitation_v"]*rs/(rc+rs), s["excitation_v"]/2
    gs = 1/(rc*rs/(rc+rs)+s["input_switch_resistance_ohm"])
    gr = 1/(rc/2+s["input_switch_resistance_ohm"])
    gd = 1/s["input_differential_resistance_ohm"]
    gc = 1/s["input_common_mode_resistance_ohm"]
    ap, am = gs+gd+gc, gr+gd+gc
    yp, ym = gs*vs-s["input_bias_a"][0], gr*vr-s["input_bias_a"][1]
    determinant = ap*am-gd*gd
    vp, vm = (yp*am+gd*ym)/determinant, (ym*ap+gd*yp)/determinant
    return s["amplifier_reference_v"]+s["amplifier_gain"]*(vp-vm)


class ReadoutFunctionals:
    def __init__(self, physics):
        self.physics = physics
        self.settings = json.loads((ROOT/"inputs/regional_readout.json").read_text())
        self.count = physics.tensor([r["gauges_per_depth"] for r in self.settings["regions"]])[None,:,None]
        calibration = [fixed_calibration(r["gauges_per_depth"],self.settings)
                       for r in self.settings["regions"]]
        self.zero = physics.tensor([c[0] for c in calibration])[None,:,None]
        self.gain = physics.tensor([c[1] for c in calibration])[None,:,None]
        # Layer order: bottom, top. Output order: q_m, q_b.
        self.transform = physics.tensor([[0.5,0.5],[-0.5,0.5]])
        self.q_matrix = torch.einsum("kl,drln->drkn",self.transform,
                                     physics.target.reshape(2,3,2,-1)).reshape(12,-1)

    def layers(self, u):
        p = self.physics
        return torch.einsum("...d,cod->...co",u,p.candidates).reshape(*u.shape[:-1],-1,2,3,2)

    def values(self, u):
        """q and F in equivalent strain, with a fixed zero-state calibration."""
        layers = self.layers(u)
        voltage = bridge_voltage(layers,self.count,self.settings)
        readout = self.physics.components((voltage-self.zero)/self.gain)
        q = torch.einsum("...d,od->...o",u,self.q_matrix).reshape(*u.shape[:-1],2,3,2)
        return q,readout,voltage

    def gradients(self, u, include_readout=True):
        """Instantaneous H_u with shape (step, output, free degree)."""
        p = self.physics
        q = self.q_matrix[None].expand(len(u),-1,-1)
        if not include_readout:
            return q[:,:,p.free].detach().cpu().numpy()
        layers = self.layers(u).detach().requires_grad_(True)
        calibrated = (bridge_voltage(layers,self.count,self.settings)-self.zero)/self.gain
        slope = torch.autograd.grad(calibrated.sum(),layers)[0]
        operators = p.candidates.reshape(-1,2,3,2,p.ndof)
        f = torch.einsum("kl,scdrl,cdrln->scdrkn",self.transform,slope,operators)
        combined = torch.cat([q,f.reshape(len(u),-1,p.ndof)],dim=1)
        return combined[:,:,p.free].detach().cpu().numpy()


class PathLinearization:
    """Sparse lower-triangular derivative of the full path force residual."""
    def __init__(self, physics, displacement, bond_scale=None, projection=None, factorize=True,
                 direction=None, *, bulk_matrix=None, compression_stiffness=None):
        p = physics
        self.nstep,self.nfree = len(displacement),len(p.free)
        if projection is not None:
            self.nfree = projection.shape[1]
        free = p.free.cpu().numpy()
        mesh = p.mesh
        u = displacement.detach()
        nodal = u.reshape(self.nstep,-1,3)
        difference = nodal[:,p.upper]-nodal[:,p.lower]
        jump = torch.einsum("qa,sfac->sfqc",p.shape,difference).flatten(1,2)
        jump = jump[None].detach().requires_grad_(True)
        onset = p.onset if bond_scale is None else p.onset*bond_scale
        compression = (p.interface_compression_stiffness if compression_stiffness is None
                       else compression_stiffness)
        if direction is None:
            traction = cohesive_path(jump,p.interface_stiffness,onset,p.final,
                                     compression_stiffness=compression)[0]
        else:
            # C_j = d_eta_j G_U along the frozen neural state tangent. Bulk,
            # wall and trestle laws are piecewise linear on the active branch.
            velocity, bond_tangent = direction
            nodes = velocity.detach().reshape(self.nstep, -1, 3)
            difference = nodes[:, p.upper]-nodes[:, p.lower]
            jump_tangent = torch.einsum("qa,sfac->sfqc", p.shape, difference).flatten(1, 2)[None]
            _, traction = torch.autograd.functional.jvp(
                lambda value, threshold: cohesive_path(value, p.interface_stiffness,
                    threshold, p.final, compression_stiffness=compression)[0], (jump, onset),
                (jump_tangent, p.onset*bond_tangent.detach()), create_graph=True)
        # Pointwise material Jacobians are obtained without a dense global AD
        # Jacobian: separate interface quadrature points do not interact.
        nquad, nshape = p.shape.shape
        width = 6*nshape
        derivative = np.zeros((self.nstep,self.nstep,len(p.area),nquad,3,3))
        for n in range(self.nstep):
            for i in range(3):
                grad = torch.autograd.grad(traction[0,n,:,i].sum(),jump,retain_graph=True)[0][0]
                derivative[n,:,:,:,i,:] = grad.detach().cpu().numpy().reshape(self.nstep,-1,nquad,3)
        shape = p.shape.cpu().numpy()
        signed_shape = np.concatenate([shape,-shape],axis=1)
        interface_conn = np.concatenate([mesh["interface_upper"],mesh["interface_lower"]],axis=1)
        wall_conn = np.concatenate([mesh["wall_substrate"],mesh["wall_cone"]],axis=1)

        def assemble(local,conn):
            dofs = (3*conn[:,:,None]+np.arange(3)).reshape(len(conn),-1)
            matrix = sparse.coo_matrix((local.reshape(-1),
                (np.broadcast_to(dofs[:,:,None],local.shape).reshape(-1),
                 np.broadcast_to(dofs[:,None,:],local.shape).reshape(-1))),shape=(p.ndof,p.ndof)).tocsr()
            return reduce(matrix)

        def reduce(matrix):
            block = matrix[free][:,free]
            return block if projection is None else projection.T@block@projection

        wall_gap = p.wall_gap(u[None])[0].cpu().numpy()
        normal = mesh["wall_normal"]
        bulk = reduce(p._bulk_matrix if bulk_matrix is None else bulk_matrix) if direction is None else None
        weights = p.surface_weights.cpu().numpy()
        self.blocks = [[None]*self.nstep for _ in range(self.nstep)]
        for n in range(self.nstep):
            for m in range(n+1):
                local = np.einsum("qa,fqij,qb,f,q->faibj",signed_shape,derivative[n,m],signed_shape,
                                  mesh["interface_area"],weights,optimize=True).reshape(-1,width,width)
                block = assemble(local,interface_conn)
                if n == m and direction is None:
                    # Match clamp_max's selected derivative at exactly zero.
                    wall_tangent = np.einsum("fq,fi,fj->fqij",wall_gap[n] <= 0,normal,normal)
                    wall_tangent *= p.settings["geometry"]["cone_contact_stiffness_pa_per_m"]
                    wall_local = np.einsum("qa,fqij,qb,f,q->faibj",signed_shape,wall_tangent,signed_shape,
                                          mesh["wall_area"],weights,optimize=True).reshape(-1,width,width)
                    contact = reduce(p.trestle_tangent(u[n].cpu().numpy()))
                    block = block+bulk+assemble(wall_local,wall_conn)+contact
                self.blocks[n][m] = block.tocsc()
                self.blocks[n][m].eliminate_zeros()
        self.factors = ([splu(self.blocks[n][n]) for n in range(self.nstep)]
                        if factorize and direction is None else [])

    def apply(self, direction):
        return np.stack([sum(self.blocks[n][m]@direction[m] for m in range(n+1))
                         for n in range(self.nstep)])

    def apply_transpose(self, values):
        """Transpose action includes every later residual's history dependence."""
        return np.stack([sum(self.blocks[n][m].T@values[n] for n in range(m, self.nstep))
                         for m in range(self.nstep)])

    def solve(self, rhs):
        """Forward path solve for local sensitivities, including prior history."""
        result = []
        for n in range(self.nstep):
            value = rhs[n].copy()
            for m in range(n):
                value -= self.blocks[n][m]@result[m]
            result.append(self.factors[n].solve(value))
        return np.stack(result)

    def adjoint(self, gradients):
        """One separate functional per step/output; retain all earlier weights.

        Returns (step, free degree, output step * instantaneous output).
        """
        nout = gradients.shape[1]
        result = [None]*self.nstep
        for n in reversed(range(self.nstep)):
            rhs = np.zeros((self.nfree,self.nstep*nout))
            rhs[:,n*nout:(n+1)*nout] = gradients[n].T
            for k in range(n+1,self.nstep):
                rhs -= self.blocks[k][n].T@result[k]
            result[n] = self.factors[n].solve(rhs,trans="T")
        return np.stack(result)


def build_adjoint_weights(physics, functionals, displacement, include_readout=True):
    """Freeze weights at a current prediction; never detach residual histories."""
    weights = []
    for u in displacement:
        linear = PathLinearization(physics,u)
        gradient = functionals.gradients(u,include_readout)
        weights.append(linear.adjoint(gradient).transpose(2,0,1))
    return physics.tensor(np.stack(weights))


def projected_correction(weights, free_force):
    return -torch.einsum("bosd,bsd->bo",weights,free_force)
