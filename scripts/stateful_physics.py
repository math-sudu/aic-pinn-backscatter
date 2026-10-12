"""Differentiable three-dimensional weak equilibrium and path history.

The spatial residual uses P1 or straight-sided P2 tetrahedra in the discrete
physics approach of FOL. It is the neural training equation. Cohesive forces
use consistent surface quadrature and an isotropic specialization of the
maximum-separation bilinear damage law.
All code uses SI units until the dimensionless network-displacement boundary.
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch import nn

from quadratic_elements import quadratic_kinematics, shape_gradients, shape_values, surface_matrix, triangle_rule

ROOT = Path(__file__).resolve().parents[1]


def tetra_kinematics(xyz, tetra):
    vertices = xyz[tetra]
    interpolation = np.concatenate([np.ones((*tetra.shape,1)),vertices],axis=-1)
    inverse = np.linalg.inv(interpolation)
    gradient = inverse[:,1:,:].transpose(0,2,1)
    volume = np.abs(np.linalg.det(vertices[:,1:]-vertices[:,0,None]))/6
    b = np.zeros((len(tetra),6,12))
    for a in range(4):
        gx,gy,gz = gradient[:,a].T
        b[:,0,3*a] = gx
        b[:,1,3*a+1] = gy
        b[:,2,3*a+2] = gz
        b[:,3,3*a] = gy
        b[:,3,3*a+1] = gx
        b[:,4,3*a+1] = gz
        b[:,4,3*a+2] = gy
        b[:,5,3*a] = gz
        b[:,5,3*a+2] = gx
    return b, volume, inverse


def assemble_stiffness(mesh, settings):
    b, volume, inverse = tetra_kinematics(mesh["xyz"],mesh["tetra"][:, :4])
    bulk = settings["bulk"]
    young = np.where(mesh["body"] == 0,bulk["top_young_pa"],bulk["substrate_young_pa"])
    nu = bulk["poisson"]
    lame = young*nu/((1+nu)*(1-2*nu))
    mu = young/(2*(1+nu))
    c = np.zeros((len(b),6,6))
    c[:,:3,:3] = lame[:,None,None]
    for i in range(3):
        c[:,i,i] += 2*mu
        c[:,i+3,i+3] = mu
    if mesh["tetra"].shape[1] == 10:
        b = quadratic_kinematics(inverse)
        ke = np.einsum("eqai,eab,eqbj,e->eij", b, c, b, volume/4, optimize=True)
    else:
        ke = np.einsum("eai,eab,ebj,e->eij",b,c,b,volume,optimize=True)
    dofs = (mesh["tetra"][:,:,None]*3+np.arange(3)).reshape(len(b),-1)
    row = np.broadcast_to(dofs[:,:,None],ke.shape).ravel()
    col = np.broadcast_to(dofs[:,None,:],ke.shape).ravel()
    size = 3*len(mesh["xyz"])
    matrix = sparse.coo_matrix((ke.ravel(),(row,col)),shape=(size,size)).tocsr()
    if "lateral_guide_operator" in mesh:
        guide = sparse.csr_matrix(mesh["lateral_guide_operator"])
        matrix += settings["geometry"]["lateral_guide_stiffness_n_per_m"]*(guide.T@guide)
    return matrix, b, volume, inverse


def cohesive_path(jump, stiffness, onset, final, detach_history=False,
                  compression_stiffness=None):
    """jump axes: (batch, physical_path_step, integration_point, xyz).

    Normal separation is positive opening. Damage never reduces compressive
    normal stiffness. History is recomputed along each candidate physical path
    every forward call; optimization iterations do not advance material time.
    detach_history exists solely to test the missing-history derivative defect.
    """
    positive = torch.cat([jump[...,:2],torch.clamp_min(jump[...,2:],0)],dim=-1)
    squared = (positive*positive).sum(dim=-1)
    effective = squared.clamp_min((onset*1e-12)**2).sqrt()
    previous = torch.zeros_like(effective[:,0])
    history = []
    for n in range(jump.shape[1]):
        prior = previous.detach() if detach_history else previous
        previous = torch.maximum(prior,effective[:,n])
        history.append(previous)
    kappa = torch.stack(history,dim=1)
    traction, damage = cohesive_response(jump, kappa, stiffness, onset, final, compression_stiffness)
    return traction, damage, kappa


def cohesive_response(jump, history, stiffness, onset, final, compression_stiffness=None):
    """Traction at an already updated maximum-separation history."""
    damage = (final*(history-onset)/(history*(final-onset))).clamp(0,1)
    factor = torch.stack([1-damage,1-damage,
                           torch.where(jump[...,2] < 0,torch.ones_like(damage),1-damage)],dim=-1)
    traction = stiffness*factor*jump
    if compression_stiffness is not None:
        # Closed surfaces have their own contact compliance. Compression does
        # not advance the opening/slip history or inherit tensile degradation.
        normal = torch.where(jump[..., 2] < 0,
                             compression_stiffness*jump[..., 2], traction[..., 2])
        traction = torch.cat([traction[..., :2], normal[..., None]], dim=-1)
    return traction, damage


def frictional_step(jump, prior, stiffness, onset, final, compression_stiffness, coefficient):
    """Cohesion plus Coulomb return on the damaged area, without dilatancy.

    State components are maximum effective separation and two plastic slips.
    The damaged-area traction is projected onto the pressure-dependent disk;
    its area fraction multiplies the returned traction, not the friction bound.
    Opening gives zero friction and updates the free tangential slip.
    """
    positive = torch.cat([jump[..., :2], jump[..., 2:].clamp_min(0)], dim=-1)
    effective = positive.square().sum(-1).clamp_min((onset*1e-12)**2).sqrt()
    history = torch.maximum(prior[..., 0], effective)
    cohesive, damage = cohesive_response(jump, history, stiffness, onset, final, compression_stiffness)
    trial = stiffness*(jump[..., :2]-prior[..., 1:])
    magnitude = trial.square().sum(-1).clamp_min((stiffness*onset*1e-12)**2).sqrt()
    limit = coefficient*compression_stiffness*(-jump[..., 2]).clamp_min(0)
    friction = trial*(limit/magnitude).clamp_max(1)[..., None]
    slip = torch.where((magnitude <= limit)[..., None], prior[..., 1:],
                       jump[..., :2]-friction/stiffness)
    traction = torch.cat([cohesive[..., :2]+damage[..., None]*friction, cohesive[..., 2:]], dim=-1)
    return traction, damage, torch.cat([history[..., None], slip], dim=-1)


def frictional_path(jump, stiffness, onset, final, compression_stiffness, coefficient,
                    detach_history=False):
    """Reconstruct both irreversible histories on each physical candidate path."""
    compression = torch.ones_like(jump[..., 2])*compression_stiffness
    thresholds = torch.ones_like(jump[..., 2])*onset
    prior = torch.zeros_like(jump[:, 0])
    tractions, damages, states = [], [], []
    for step in range(jump.shape[1]):
        previous = prior.detach() if detach_history else prior
        traction, damage, prior = frictional_step(jump[:, step], previous, stiffness,
            thresholds[:, step], final, compression[:, step], coefficient)
        tractions.append(traction)
        damages.append(damage)
        states.append(prior)
    return tuple(torch.stack(values, dim=1) for values in (tractions, damages, states))


def cohesive_current_tangent(jump, prior_history, stiffness, onset, final,
                             compression_stiffness):
    """Current-step tangent at fixed prior history; loading-side at a maximum."""
    positive = np.concatenate([jump[..., :2], np.maximum(jump[..., 2:], 0)], axis=-1)
    effective = np.sqrt(np.maximum(np.sum(positive**2, axis=-1), (onset*1e-12)**2))
    history = np.maximum(prior_history, effective)
    damage = np.clip(final*(history-onset)/(history*(final-onset)), 0, 1)
    tangent = np.zeros((*jump.shape[:-1], 3, 3))
    for i in range(3):
        tangent[..., i, i] = stiffness*(1-damage)
    tangent[..., 2, 2] = np.where(jump[..., 2] < 0, compression_stiffness, tangent[..., 2, 2])
    loading = (effective >= prior_history) & (history > onset) & (history < final)
    rate = np.where(loading, final*onset/((final-onset)*history**2*effective), 0.)
    tangent -= stiffness*rate[..., None, None]*positive[..., :, None]*positive[..., None, :]
    return tangent


def line_average_operator(mesh, inverse, start, end, direction):
    """Integrate host epsilon_dd along a fixed straight segment exactly.

    Intersect the segment with each top tetrahedron using barycentric affine
    inequalities. Midpoint integration is exact for P1's constant and P2's
    linear strain on each interval. Split coincident face intervals once.
    """
    selected = np.flatnonzero(mesh["body"] == 0)
    inv = inverse[selected]
    lam0 = np.einsum("i,eij->ej",np.r_[1,start],inv)
    delta = np.einsum("i,eij->ej",np.r_[0,np.asarray(end)-start],inv)
    low, high = np.zeros(len(inv)), np.ones(len(inv))
    valid = np.ones(len(inv),bool)
    for k in range(4):
        slope, value = delta[:,k], lam0[:,k]
        varying = np.abs(slope) > 1e-12
        cut = np.divide(-value,slope,out=np.zeros_like(value),where=varying)
        low = np.where(slope > 1e-12,np.maximum(low,cut),low)
        high = np.where(slope < -1e-12,np.minimum(high,cut),high)
        valid &= varying | (value >= -1e-10)
    valid &= high-low > 1e-10
    indices, low, high = selected[valid],low[valid],high[valid]
    breaks = np.unique(np.round(np.r_[0,1,low,high],12))
    result = np.zeros(3*len(mesh["xyz"]))
    for left,right in zip(breaks[:-1],breaks[1:]):
        if right-left < 1e-10:
            continue
        mid = (left+right)/2
        covering = np.flatnonzero((low < mid+1e-10) & (high > mid-1e-10))
        if not len(covering):
            raise ValueError(f"Observation segment leaves the mesh at fraction {mid}")
        element = indices[covering[0]]
        gradient = inverse[element, 1:, :].T
        if mesh["tetra"].shape[1] == 10:
            point = np.asarray(start)+mid*(np.asarray(end)-start)
            bary = np.r_[1, point]@inverse[element]
            gradient = shape_gradients(bary, gradient)
        np.add.at(result,3*mesh["tetra"][element]+direction, (right-left)*gradient[:, direction])
    return result


def observation_operators(mesh, settings, inverse):
    obs = settings["observations"]
    circuit = json.loads((ROOT/"inputs/regional_readout.json").read_text(encoding="utf-8"))
    targets, candidates = [], []
    # Each output has direction, region, depth axes; q_m/q_b follow downstream.
    for direction in obs["directions"]:
        for region,c in zip(obs["regions_m"],circuit["regions"]):
            left,right = region
            centres = left+(right-left)*(np.arange(c["gauges_per_depth"])+0.5)/c["gauges_per_depth"]
            for z in obs["depths_m"]:
                def segment(a,b):
                    start,end = np.array([0.,0.,z]),np.array([0.,0.,z])
                    start[direction],end[direction] = a,b
                    return line_average_operator(mesh,inverse,start,end,direction)
                targets.append(segment(left,right))
                rows = []
                for candidate in obs["candidates"]:
                    length,shift = candidate["length_m"],candidate["shift_m"]
                    if centres[0]+shift-length/2 < left-1e-12 or centres[-1]+shift+length/2 > right+1e-12:
                        raise ValueError("Gauge support leaves the fixed target region")
                    rows.append(np.mean([segment(x+shift-length/2,x+shift+length/2) for x in centres],axis=0))
                candidates.append(rows)
    return np.stack(targets),np.stack(candidates).transpose(1,0,2)


class PathPhysics:
    def __init__(self, mesh, settings, device="cpu"):
        self.mesh,self.settings = mesh,settings
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.scale = settings["loading"]["displacement_scale_m"]
        self.path = self.tensor(settings["loading"]["path"])[1:]
        self.ndof = 3*len(mesh["xyz"])
        constrained = np.union1d(mesh["fixed_dofs"],mesh["driven_dofs"])
        self.free = self.tensor(np.setdiff1d(np.arange(self.ndof),constrained),torch.long)
        self.driven = self.tensor(mesh["driven_dofs"],torch.long)
        self.fixed = self.tensor(mesh["fixed_dofs"],torch.long)
        stiffness,b,volume,inverse = assemble_stiffness(mesh,settings)
        coo = stiffness.tocoo()
        self.stiffness = torch.sparse_coo_tensor(self.tensor(np.stack([coo.row,coo.col]),torch.long),
            self.tensor(coo.data),coo.shape,device=self.device).coalesce()
        self.diagonal = self.tensor(stiffness.diagonal())
        self.b,self.volume,self.inverse = b,volume,inverse
        self.upper = self.tensor(mesh["interface_upper"],torch.long)
        self.lower = self.tensor(mesh["interface_lower"],torch.long)
        self.area = self.tensor(mesh["interface_area"])
        self.wall_cone = self.tensor(mesh["wall_cone"],torch.long)
        self.wall_substrate = self.tensor(mesh["wall_substrate"],torch.long)
        self.wall_area = self.tensor(mesh["wall_area"])
        self.wall_normal = self.tensor(mesh["wall_normal"])
        if mesh["tetra"].shape[1] == 10:
            bary, weights = triangle_rule()
            self.shape = self.tensor(shape_values(bary))
            self.surface_weights = self.tensor(weights)
            self.trestle_faces = self.tensor(mesh["trestle_faces"], torch.long)
            self.trestle_face_area = self.tensor(mesh["trestle_face_area"])
        else:
            self.shape = self.tensor([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
            self.surface_weights = self.tensor(np.full(3, 1/3))
            self.trestle_faces = None
        self.trestle_area = self.tensor(mesh["trestle_area"])
        interface = settings["interface"]
        self.interface_stiffness = interface["stiffness_pa_per_m"]
        self.interface_compression_stiffness = interface.get(
            "compression_stiffness_pa_per_m", self.interface_stiffness)
        self.interface_friction_coefficient = 0.
        self.interface_coordinate_scale = None
        self.onset = interface["strength_pa"]/self.interface_stiffness
        self.final = 2*interface["fracture_energy_j_per_m2"]/interface["strength_pa"]
        if not self.final > self.onset:
            raise ValueError("Bilinear interface needs positive softening energy")
        target,candidates = observation_operators(mesh,settings,inverse)
        self.target,self.candidates = self.tensor(target),self.tensor(candidates)
        self.elastic_metric = None
        self._bulk_matrix = stiffness

    def interface_path(self, jump, compression_stiffness=None, *, onset=None, detach_history=False):
        # A fixed pointwise coordinate projection removes released shear from
        # both the stored energy and damage/slip history. Tractions transform
        # back by its transpose, preserving virtual work.
        scale = self.interface_coordinate_scale
        if scale is not None:
            jump = jump*scale
        compression = (self.interface_compression_stiffness if compression_stiffness is None
                       else compression_stiffness)
        onset = self.onset if onset is None else onset
        if self.interface_friction_coefficient:
            result = frictional_path(jump, self.interface_stiffness, onset, self.final,
                compression, self.interface_friction_coefficient, detach_history)
        else:
            result = cohesive_path(jump, self.interface_stiffness, onset, self.final,
                                   detach_history, compression)
        return result if scale is None else (result[0]*scale, *result[1:])

    def interface_step(self, jump, prior, compression_stiffness):
        scale = self.interface_coordinate_scale
        if scale is not None:
            jump = jump*scale
        if self.interface_friction_coefficient:
            result = frictional_step(jump, prior, self.interface_stiffness, self.onset,
                self.final, compression_stiffness, self.interface_friction_coefficient)
        else:
            positive = torch.cat([jump[..., :2], jump[..., 2:].clamp_min(0)], dim=-1)
            effective = positive.square().sum(-1).clamp_min((self.onset*1e-12)**2).sqrt()
            history = torch.maximum(prior, effective)
            traction, damage = cohesive_response(jump, history, self.interface_stiffness,
                self.onset, self.final, compression_stiffness)
            result = traction, damage, history
        return result if scale is None else (result[0]*scale, *result[1:])

    def elastic_stiffness(self):
        """Fixed intact elastic operator for the residual dual-energy norm.

        Factorization supplies a loss metric only. It neither produces labels
        nor solves the nonlinear loading path. Histories and their derivatives
        remain inside the learned path residual.
        """
        area = self.mesh["interface_area"]
        shape, weights = self.shape.cpu().numpy(), self.surface_weights.cpu().numpy()
        mass = area[:,None,None]*np.einsum("qa,q,qb->ab", shape, weights, shape)[None]
        pair = np.concatenate([np.concatenate([mass,-mass],axis=2),
                               np.concatenate([-mass,mass],axis=2)],axis=1)
        width = 6*shape.shape[1]
        local = np.einsum("eab,ij->eaibj",pair,np.eye(3)).reshape(-1,width,width)*self.interface_stiffness
        conn = np.concatenate([self.mesh["interface_upper"],self.mesh["interface_lower"]],axis=1)
        matrix = self._bulk_matrix+surface_matrix(conn, local, self.ndof)
        matrix += self.trestle_tangent(np.zeros(self.ndof))
        wall_mass = self.mesh["wall_area"][:,None,None]*np.einsum("qa,q,qb->ab", shape, weights, shape)[None]
        wall_pair = np.concatenate([np.concatenate([wall_mass,-wall_mass],axis=2),
                                    np.concatenate([-wall_mass,wall_mass],axis=2)],axis=1)
        normals = self.mesh["wall_normal"]
        normal_tensor = normals[:,:,None]*normals[:,None,:]
        local = np.einsum("eab,eij->eaibj",wall_pair,normal_tensor).reshape(-1,width,width)
        local *= self.settings["geometry"]["cone_contact_stiffness_pa_per_m"]
        conn = np.concatenate([self.mesh["wall_substrate"],self.mesh["wall_cone"]],axis=1)
        matrix += surface_matrix(conn, local, self.ndof)
        return matrix

    def prepare_elastic_metric(self):
        matrix = self.elastic_stiffness()
        free = self.free.cpu().numpy()
        reduced = matrix[free][:,free].toarray()
        self.elastic_metric = torch.linalg.cholesky(self.tensor(reduced))

    def tensor(self, value, dtype=None):
        return torch.as_tensor(value,dtype=dtype or self.dtype,device=self.device)

    def displacements(self, free_values, amplitude):
        # Physical displacement, with strong Dirichlet conditions.
        u = free_values.new_zeros((len(amplitude),len(self.path),self.ndof))
        u[:,:,self.free] = free_values*self.scale
        u[:,:,self.driven] = amplitude[:,None,None]*self.path[None,:,None]*self.scale
        return u

    def forces(self, u, detach_history=False, bond_scale=None, compression_stiffness=None,
               *, return_components=False):
        flat = u.reshape(-1,self.ndof)
        bulk = torch.sparse.mm(self.stiffness,flat.T).T.reshape(u.shape)
        nodal = u.reshape(*u.shape[:-1],-1,3)
        difference = nodal[:,:,self.upper,:]-nodal[:,:,self.lower,:]
        jump = torch.einsum("qa,bsfac->bsfqc",self.shape,difference)
        # Spatial strength and fracture energy scale together at fixed K.
        # Their ratio leaves final separation unchanged; onset varies locally.
        onset = self.onset if bond_scale is None else self.onset*bond_scale
        traction, damage, interface_state = self.interface_path(jump.flatten(2, 3),
            compression_stiffness, onset=onset, detach_history=detach_history)
        history = interface_state[..., 0] if self.interface_friction_coefficient else interface_state
        traction = traction.reshape(jump.shape)
        nodal_force = torch.einsum("qa,bsfqc,f,q->bsfac",self.shape,traction,self.area,self.surface_weights)
        interface_force = torch.zeros_like(nodal)
        interface_force.index_add_(2,self.upper.flatten(),nodal_force.flatten(2,3))
        interface_force.index_add_(2,self.lower.flatten(),-nodal_force.flatten(2,3))
        if return_components:
            cohesive_force = interface_force.clone()
        wall_gap = self.wall_gap(u)
        wall_traction = (wall_gap.clamp_max(0)[...,None]*self.wall_normal[None,None,:,None,:]
                         *self.settings["geometry"]["cone_contact_stiffness_pa_per_m"])
        wall_force = torch.einsum("qa,bsfqc,f,q->bsfac",self.shape,wall_traction,self.wall_area,self.surface_weights)
        interface_force.index_add_(2,self.wall_substrate.flatten(),wall_force.flatten(2,3))
        interface_force.index_add_(2,self.wall_cone.flatten(),-wall_force.flatten(2,3))
        if return_components:
            cone_force = torch.zeros_like(nodal)
            cone_force.index_add_(2,self.wall_substrate.flatten(),wall_force.flatten(2,3))
            cone_force.index_add_(2,self.wall_cone.flatten(),-wall_force.flatten(2,3))
        if self.trestle_faces is None:
            contact_z = nodal[...,2].clamp_max(0)*self.trestle_area
        else:
            gap = torch.einsum("qa,bsfa->bsfq", self.shape, nodal[:, :, self.trestle_faces, 2])
            local = torch.einsum("qa,bsfq,f,q->bsfa", self.shape, gap.clamp_max(0),
                                 self.trestle_face_area, self.surface_weights)
            contact_z = torch.zeros_like(nodal[..., 2])
            contact_z.index_add_(2, self.trestle_faces.flatten(), local.flatten(2, 3))
        contact_z = contact_z*self.settings["geometry"]["trestle_stiffness_pa_per_m"]
        contact = torch.stack([torch.zeros_like(contact_z),torch.zeros_like(contact_z),contact_z],dim=-1)
        result = bulk+(interface_force+contact).flatten(-2),damage,history
        if return_components:
            return (*result, dict(bulk_and_guides=bulk, interface=cohesive_force.flatten(-2),
                cone_contact=cone_force.flatten(-2), trestle=contact.flatten(-2),
                interface_jump=jump, interface_traction=traction,
                interface_state=interface_state,
                cone_gap=wall_gap, cone_traction=wall_traction))
        return result

    def interface_closure_tangent(self, displacement):
        """Closed-normal correction to the intact opening/shear operator."""
        difference = self.interface_compression_stiffness-self.interface_stiffness
        if difference == 0:
            return sparse.csr_matrix((self.ndof, self.ndof))
        upper, lower = self.mesh["interface_upper"], self.mesh["interface_lower"]
        shape, weights = self.shape.cpu().numpy(), self.surface_weights.cpu().numpy()
        nodal = np.asarray(displacement).reshape(-1, 3)
        gap = np.einsum("qa,fa->fq", shape, nodal[upper, 2]-nodal[lower, 2])
        signed = np.concatenate([shape, -shape], axis=1)
        mass = np.einsum("qa,fq,qb,f,q->fab", signed, gap < 0, signed,
                         self.mesh["interface_area"]*difference, weights)
        width = 6*shape.shape[1]
        local = np.einsum("fab,ij->faibj", mass, np.diag([0., 0., 1.])).reshape(-1, width, width)
        return surface_matrix(np.concatenate([upper, lower], axis=1), local, self.ndof)

    def interface_history_correction_tangent(self, displacement, prior_history):
        """Full cohesive tangent minus the intact opening/shear matrix."""
        upper, lower = self.mesh["interface_upper"], self.mesh["interface_lower"]
        shape, weights = self.shape.cpu().numpy(), self.surface_weights.cpu().numpy()
        nodal = np.asarray(displacement).reshape(-1, 3)
        jump = np.einsum("qa,fac->fqc", shape, nodal[upper]-nodal[lower])
        tangent = cohesive_current_tangent(jump, np.asarray(prior_history).reshape(jump.shape[:2]),
                                           self.interface_stiffness, self.onset, self.final,
                                           self.interface_compression_stiffness)
        tangent -= self.interface_stiffness*np.eye(3)
        signed = np.concatenate([shape, -shape], axis=1)
        width = 6*shape.shape[1]
        local = np.einsum("qa,fqij,qb,f,q->faibj", signed, tangent, signed,
                          self.mesh["interface_area"], weights, optimize=True).reshape(-1, width, width)
        return surface_matrix(np.concatenate([upper, lower], axis=1), local, self.ndof)

    def trestle_tangent(self, displacement):
        stiffness = self.settings["geometry"]["trestle_stiffness_pa_per_m"]
        if self.trestle_faces is None:
            contact = np.zeros(self.ndof)
            contact[2::3] = (displacement[2::3] <= 0)*self.mesh["trestle_area"]*stiffness
            return sparse.diags(contact, format="csr")
        conn = self.mesh["trestle_faces"]
        shape, weights = self.shape.cpu().numpy(), self.surface_weights.cpu().numpy()
        gap = np.einsum("qa,fa->fq", shape, displacement.reshape(-1, 3)[conn, 2])
        mass = np.einsum("qa,fq,qb,f,q->fab", shape, gap <= 0, shape,
                         self.mesh["trestle_face_area"]*stiffness, weights)
        normal = np.diag([0., 0., 1.])
        local = np.einsum("fab,ij->faibj", mass, normal).reshape(-1, 3*conn.shape[1], 3*conn.shape[1])
        return surface_matrix(conn, local, self.ndof)

    def wall_gap(self,u):
        nodal = u.reshape(*u.shape[:-1],-1,3)
        jump = nodal[:,:,self.wall_substrate,:]-nodal[:,:,self.wall_cone,:]
        return torch.einsum("qa,bsfac,fc->bsfq",self.shape,jump,self.wall_normal)

    def residual(self, free_values, amplitude, detach_history=False):
        u = self.displacements(free_values,amplitude)
        force,_,_ = self.forces(u,detach_history)
        free_force = force[:,:,self.free]
        return self.whiten_force(free_force)

    def whiten_force(self, free_force):
        """Apply the existing residual metric to free-degree physical forces."""
        if self.elastic_metric is None:
            return free_force/(self.diagonal[self.free]*self.scale)
        whitened = torch.linalg.solve_triangular(self.elastic_metric,
            free_force.flatten(0,1).T,upper=False).T.reshape(free_force.shape)
        regions = self.settings["observations"]["regions_m"]
        length_scale = max(r[1] for r in regions)-min(r[0] for r in regions)
        energy_scale = self.settings["bulk"]["top_young_pa"]*length_scale*self.scale**2
        return whitened/np.sqrt(energy_scale)

    def observations(self,u):
        target = torch.einsum("...d,od->...o",u,self.target).reshape(*u.shape[:-1],2,3,2)
        candidate = torch.einsum("...d,cod->...co",u,self.candidates).reshape(*u.shape[:-1],len(self.candidates),2,3,2)
        return self.components(target),self.components(candidate)

    @staticmethod
    def components(layers):
        return torch.stack([(layers[...,1]+layers[...,0])/2,
                            (layers[...,1]-layers[...,0])/2],dim=-1)


class PathNetwork(nn.Module):
    """Amplitude -> complete displacement path; history lives in PathPhysics."""
    def __init__(self,physics,width):
        super().__init__()
        self.nstep,self.nfree = len(physics.path),len(physics.free)
        self.layers = nn.Sequential(nn.Linear(1,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),
                                    nn.Linear(width,self.nstep*self.nfree))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
        self.to(device=physics.device,dtype=physics.dtype)

    def forward(self,amplitude):
        output = self.layers((amplitude-1)[:,None]).reshape(-1,self.nstep,self.nfree)
        return amplitude[:,None,None]*output


class MetricPathNetwork(PathNetwork):
    """Normalized amplitude and invertible intact-elastic output coordinates."""

    def __init__(self, physics, width):
        super().__init__(physics, width)
        self.factor = physics.elastic_metric
        self.amplitude_scale = physics.settings["loading"]["amplitude_derivative_scale"]
        regions = physics.settings["observations"]["regions_m"]
        span = max(r[1] for r in regions)-min(r[0] for r in regions)
        self.coordinate_scale = np.sqrt(physics.settings["bulk"]["top_young_pa"]*span)

    def forward(self, amplitude):
        raw = self.layers(((amplitude-1)/self.amplitude_scale)[:, None])
        raw = amplitude[:, None, None]*raw.reshape(-1, self.nstep, self.nfree)
        values = torch.linalg.solve_triangular(self.factor.T, raw.flatten(0, 1).T, upper=True)
        return values.T.reshape(raw.shape)*self.coordinate_scale

    def load_direct_state(self, state):
        """Preserve a PathNetwork's values and derivatives under this basis change.

        Input normalization multiplies the first weight by the amplitude scale.
        Each last-layer block becomes L^T W/sqrt(E_top*span), with the same
        transform on its bias. The forward triangular solve then cancels it.
        """
        self.load_state_dict(state)
        with torch.no_grad():
            self.layers[0].weight.mul_(self.amplitude_scale)
            last = self.layers[-1]
            weight = last.weight.reshape(self.nstep, self.nfree, -1)
            bias = last.bias.reshape(self.nstep, self.nfree)
            last.weight.copy_((torch.einsum("ji,sjh->sih", self.factor, weight)
                               / self.coordinate_scale).reshape_as(last.weight))
            last.bias.copy_((torch.einsum("ji,sj->si", self.factor, bias)
                             / self.coordinate_scale).reshape_as(last.bias))


def load_problem(device="cpu"):
    settings = json.loads((ROOT/"inputs/stateful_pilot.json").read_text(encoding="utf-8"))
    with np.load(ROOT/"data/processed/stateful_pilot/mesh.npz") as source:
        mesh = dict(source)
    return PathPhysics(mesh,settings,device)
