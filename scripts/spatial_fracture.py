"""Nonlocal Rankine tensile history on the existing P2 volume kinematics.

Each configured body has its own history. Tensile effective stresses are degraded;
compressive principal stresses recover the elastic response on closure.
The fixed interaction radius and softening conversion length are distinct
conditional material inputs, not tetrahedron or quadrature-cell dimensions.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch


class TensileSpectrum(torch.autograd.Function):
    """Positive stress tensor and largest eigenvalue with stable first derivatives.

    Divided differences avoid eigenvector-gradient singularities at repeated
    eigenvalues. At a repeated maximum use its mean spectral projector; at a
    zero eigenvalue use the centred positive-part derivative. These choices
    specify first-order generalized derivatives at nonsmooth points.
    """

    @staticmethod
    def forward(ctx, stress):
        values, vectors = torch.linalg.eigh(stress)
        ctx.save_for_backward(values, vectors)
        positive = (vectors*values.clamp_min(0)[..., None, :])@vectors.transpose(-1, -2)
        return positive, values[..., -1]

    @staticmethod
    def backward(ctx, positive_gradient, maximum_gradient):
        values, vectors = ctx.saved_tensors
        left, right = values[..., :, None], values[..., None, :]
        difference = left-right
        tolerance = 32*torch.finfo(values.dtype).eps*values.abs().amax(-1)[..., None, None]
        separated = difference.abs() > tolerance
        slope = (left.clamp_min(0)-right.clamp_min(0))/torch.where(
            separated, difference, torch.ones_like(difference))
        midpoint = (left+right)/2
        repeated = torch.where(midpoint > 0, 1., torch.where(midpoint < 0, 0., .5))
        slope = torch.where(separated, slope, repeated)
        gradient = values.new_zeros((*values.shape[:-1], 3, 3))
        if positive_gradient is not None:
            symmetric = (positive_gradient+positive_gradient.transpose(-1, -2))/2
            local = vectors.transpose(-1, -2)@symmetric@vectors
            gradient = vectors@(slope*local)@vectors.transpose(-1, -2)
        if maximum_gradient is not None:
            tied = (values-values[..., -1:]).abs() <= tolerance[..., 0]
            weight = tied.to(values.dtype)/tied.sum(-1, keepdim=True)
            projector = (vectors*weight[..., None, :])@vectors.transpose(-1, -2)
            gradient = gradient+maximum_gradient[..., None, None]*projector
        return gradient


def elastic_stress(strain, young, poisson):
    """Engineering strain order xx, yy, zz, xy, yz, xz; tensor stress output."""
    components = strain*strain.new_tensor([1., 1., 1., .5, .5, .5])
    tensor = components[..., [[0, 3, 5], [3, 1, 4], [5, 4, 2]]]
    mu = young/(2*(1+poisson))
    lame = young*poisson/((1+poisson)*(1-2*poisson))
    return (2*mu[..., None, None]*tensor + lame[..., None, None]
            *strain[..., :3].sum(-1)[..., None, None]*torch.eye(3, dtype=strain.dtype, device=strain.device))


def stress_components(stress):
    return stress[..., [0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]]


def nonlocal_weights(points, volumes, radius):
    """Jirasek (2007), equations (9), (37)-(38); normalized on the FRC body."""
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("The material interaction radius must be positive")
    tree = cKDTree(points)
    distance = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    weight = (1-(distance.data/radius)**2)**2*np.asarray(volumes)[distance.col]
    matrix = sparse.coo_matrix((weight, (distance.row, distance.col)), distance.shape).tocsr()
    matrix.eliminate_zeros()
    return sparse.diags(1/np.asarray(matrix.sum(axis=1)).ravel())@matrix


def average_field(values, weights):
    flat = values.reshape(-1, values.shape[-1])
    return torch.sparse.mm(weights, flat.T).T.reshape(values.shape)


def tensile_stress_path(strain, young, poisson, law, softening_length, weights):
    """Full space/time graph; strain axes are batch, time, point, component."""
    modulus = torch.as_tensor(young, dtype=strain.dtype, device=strain.device)
    if modulus.ndim == 1:
        modulus = modulus[:, None, None]
    effective = elastic_stress(strain, modulus, poisson)
    positive, maximum = TensileSpectrum.apply(effective)
    equivalent = maximum.clamp_min(0)/modulus
    averaged = average_field(equivalent, weights)
    # The scalar law supplies the common history and stable secant factor.
    # Its opening is an equivalent constitutive coordinate, not a crack map.
    material_modulus = modulus[:, 0] if modulus.ndim == 3 else modulus
    _, damage, history, opening = law.stress_path(averaged, material_modulus, softening_length)
    onset = law.tensile_strength_pa/modulus
    factor = torch.where(history > onset,
        law.strain_envelope(history, modulus, softening_length)/(modulus*history.clamp_min(onset)),
        torch.ones_like(history))
    nominal = effective-positive+factor[..., None, None]*positive
    return stress_components(nominal), dict(damage=damage, history=history,
        equivalent_strain=equivalent, nonlocal_strain=averaged, equivalent_opening_m=opening)


class BodyFracture:
    def __init__(self, physics, law, radius_m, softening_length_m, *, body=0):
        p = self.physics = physics
        self.law, self.softening_length = law, softening_length_m
        selected = np.flatnonzero(p.mesh["body"] == body)
        self.elements = selected
        self.conn = p.tensor(p.mesh["tetra"][selected], torch.long)
        b = p.b[selected]
        if b.ndim == 3:
            b = b[:, None]
            barycentric = np.full((1, 4), .25)
        else:
            a, c = (5+3*np.sqrt(5))/20, (5-np.sqrt(5))/20
            barycentric = np.full((4, 4), c)
            np.fill_diagonal(barycentric, a)
        self.b = p.tensor(b)
        self.nquad = b.shape[1]
        self.measure = p.tensor(p.volume[selected]/self.nquad)
        points = np.einsum("qa,eac->eqc", barycentric, p.mesh["xyz"][p.mesh["tetra"][selected, :4]])
        self.points = points.reshape(-1, 3)
        self.volumes = np.repeat(p.volume[selected]/self.nquad, self.nquad)
        self.averaging = nonlocal_weights(self.points, self.volumes, radius_m).tocsr()
        coo = self.averaging.tocoo()
        self.weights = torch.sparse_coo_tensor(p.tensor(np.stack([coo.row, coo.col]), torch.long),
            p.tensor(coo.data), coo.shape).coalesce()

    def strains(self, displacement):
        nodal = displacement.reshape(*displacement.shape[:2], -1, 3)
        local = nodal[:, :, self.conn].flatten(-2)
        return torch.einsum("eqij,bsej->bseqi", self.b, local).flatten(2, 3)

    def stress_path(self, displacement, young):
        strain = self.strains(displacement)
        return tensile_stress_path(strain, young, self.physics.settings["bulk"]["poisson"],
            self.law, self.softening_length, self.weights)

    def force(self, displacement, young):
        stress, state = self.stress_path(displacement, young)
        stress = stress.reshape(*displacement.shape[:2], len(self.conn), self.nquad, 6)
        force = torch.einsum("eqij,bseqi,e->bsej", self.b, stress, self.measure)
        result = torch.zeros((*displacement.shape[:2], self.physics.ndof//3, 3),
                             dtype=displacement.dtype, device=displacement.device)
        result.index_add_(2, self.conn.flatten(), force.reshape(*displacement.shape[:2], -1, 3))
        return result.flatten(-2), state

    def elastic_matrix(self, young):
        """Undamaged operator on this body's quadrature, without guide forces."""
        from quadratic_elements import surface_matrix

        p = self.physics
        stiffness = stress_components(elastic_stress(p.tensor(np.eye(6)), p.tensor(young),
            p.settings["bulk"]["poisson"])).numpy().T
        b = self.b.numpy()
        local = np.einsum("eqai,ab,eqbj,e->eij", b, stiffness, b,
                          self.measure.numpy(), optimize=True)
        return surface_matrix(self.conn.numpy(), local, p.ndof).tocsr()
