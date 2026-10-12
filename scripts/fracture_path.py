"""Residual tensile crack-band history and work-conjugate load control.

The scalar material law is a normal-tension component, not a multiaxial
concrete model. Parameters define local stress versus crack opening. Neither
specimen force nor CMOD may be substituted for those quantities.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch


@dataclass(frozen=True)
class ResidualTensionLaw:
    """Bilinear sigma(w) as in Lofgren et al. (2005), Figure 12; SI units."""

    tensile_strength_pa: float
    a1_per_m: float
    a2_per_m: float
    b2: float

    def __post_init__(self):
        values = [self.tensile_strength_pa, self.a1_per_m, self.a2_per_m, self.b2]
        if not np.isfinite(values).all() or not (
                self.tensile_strength_pa > 0 and self.a1_per_m > self.a2_per_m > 0
                and self.a2_per_m / self.a1_per_m < self.b2 < 1):
            raise ValueError("Tensile law needs a positive, continuous two-slope softening envelope")

    @property
    def opening_knots_m(self):
        return np.array([0., (1-self.b2)/(self.a1_per_m-self.a2_per_m), self.b2/self.a2_per_m])

    @property
    def stress_knots_pa(self):
        middle = self.tensile_strength_pa*(1-self.a1_per_m*self.opening_knots_m[1])
        return np.array([self.tensile_strength_pa, middle, 0.])

    @property
    def fracture_energy_j_per_m2(self):
        return float(np.trapezoid(self.stress_knots_pa, self.opening_knots_m))

    def traction(self, opening):
        """Monotonic envelope; negative opening is outside this tensile law."""
        first = self.tensile_strength_pa*(1-self.a1_per_m*opening)
        second = self.tensile_strength_pa*(self.b2-self.a2_per_m*opening)
        return torch.where(opening <= self.opening_knots_m[1], first, second).clamp_min(0)

    def strain_envelope(self, strain, young_pa, band_length_m):
        """Invert epsilon = sigma(w)/E + w/h without changing fracture work.

        A nonmonotonic local epsilon(w) needs an independent opening variable;
        it cannot be represented by this strain-driven component.
        """
        young = torch.as_tensor(young_pa, dtype=strain.dtype, device=strain.device)
        length = torch.as_tensor(band_length_m, dtype=strain.dtype, device=strain.device)
        if torch.any(young <= 0) or torch.any(length <= 0):
            raise ValueError("Young modulus and crack-band length must be positive")
        stress = strain.new_tensor(self.stress_knots_pa)
        opening = strain.new_tensor(self.opening_knots_m)
        knots = stress/young[..., None] + opening/length[..., None]
        intervals = knots[..., 1:]-knots[..., :-1]
        if torch.any(intervals <= 0):
            raise ValueError("Local crack-band snap-back: use an independent crack opening or a shorter band")
        result = young*strain
        for i in range(2):
            slope = (stress[i+1]-stress[i])/intervals[..., i]
            result = torch.where(strain > knots[..., i],
                                 stress[i]+slope*(strain-knots[..., i]), result)
        return torch.where(strain >= knots[..., -1], torch.zeros_like(result), result).clamp_min(0)

    def stress_path(self, strain, young_pa, band_length_m):
        """Axes (batch, physical time, ...); secant unload and closed compression.

        Rebuild history on every call. Derivatives at unloading states retain
        their dependence on earlier maxima; optimizer calls never advance time.
        The closure/reload rule is a modelling assumption, not WST cyclic data.
        """
        young = torch.as_tensor(young_pa, dtype=strain.dtype, device=strain.device)
        length = torch.as_tensor(band_length_m, dtype=strain.dtype, device=strain.device)
        history = torch.zeros_like(strain[:, 0])
        onset = self.tensile_strength_pa/young
        stresses, damages, maxima, openings = [], [], [], []
        for current in strain.unbind(dim=1):
            history = torch.maximum(history, current.clamp_min(0))
            envelope = self.strain_envelope(history, young, length)
            secant = torch.where(history > onset, envelope/history.clamp_min(onset), young)
            damage = 1-secant/young
            stress = torch.where(current < 0, young*current, secant*current)
            stresses.append(stress)
            damages.append(damage)
            maxima.append(history)
            openings.append(length*(current-stress/young).clamp_min(0))
        return tuple(torch.stack(rows, dim=1) for rows in (stresses, damages, maxima, openings))


@dataclass(frozen=True)
class TensileCapLaw:
    """Conditional tensile plateau with secant unloading, for causal diagnosis.

    This has no finite fracture energy and is not a calibrated concrete law.
    The cap acts on the nonlocal driving strain in the spatial formulation.
    """

    tensile_strength_pa: float

    def __post_init__(self):
        if not np.isfinite(self.tensile_strength_pa) or self.tensile_strength_pa <= 0:
            raise ValueError("The tensile cap must be positive and finite")

    def strain_envelope(self, strain, young_pa, band_length_m):
        return (young_pa*strain).clamp(max=self.tensile_strength_pa)

    def stress_path(self, strain, young_pa, band_length_m):
        young = torch.as_tensor(young_pa, dtype=strain.dtype, device=strain.device)
        history = torch.zeros_like(strain[:, 0])
        onset = self.tensile_strength_pa/young
        stresses, damages, maxima, openings = [], [], [], []
        for current in strain.unbind(dim=1):
            history = torch.maximum(history, current.clamp_min(0))
            factor = torch.where(history > onset, onset/history.clamp_min(onset),
                                 torch.ones_like(history))
            stress = torch.where(current < 0, young*current, young*factor*current)
            stresses.append(stress)
            damages.append(1-factor)
            maxima.append(history)
            openings.append(band_length_m*(current-stress/young).clamp_min(0))
        return tuple(torch.stack(rows, dim=1) for rows in (stresses, damages, maxima, openings))


def displacement_constraint(displacement, load_n, operator, target_m, *,
                            reference_displacement, reference_load_n=0., compliance_m_per_n=0.):
    """Apparatus increment: b.(u-u0) + C*(F-F0) - prescribed motion.

    b must measure motion conjugate to the unknown applied force, relative to
    its actual reaction frame. A local LVDT average or the arbitrary gauge in
    measured-traction replay is not that registration.
    """
    if compliance_m_per_n < 0:
        raise ValueError("Passive series compliance must be nonnegative")
    return ((displacement-reference_displacement) @ operator
            + compliance_m_per_n*(load_n-reference_load_n)-target_m)


def displacement_control_correction(tangent, load_derivative, operator,
                                    compliance_m_per_n, residual_n, constraint_m):
    """Solve the bordered Newton equation with the applied load unknown.

    load_derivative is dR/dF, including any genuinely load-dependent boundary.
    Measured rod/support tractions cannot remain fixed as independent labels
    when the unknown jack force changes: that would fix its global resultant.
    """
    if compliance_m_per_n < 0:
        raise ValueError("Passive series compliance must be nonnegative")
    column = sparse.csc_matrix(np.asarray(load_derivative).reshape(-1, 1))
    row = sparse.csc_matrix(np.asarray(operator).reshape(1, -1))
    matrix = sparse.bmat([[sparse.csc_matrix(tangent), column],
                          [row, sparse.csc_matrix([[compliance_m_per_n]])]], format="csc")
    correction = spsolve(matrix, -np.r_[residual_n, constraint_m])
    if not np.isfinite(correction).all():
        raise RuntimeError("The selected displacement control has a singular tangent")
    return correction[:-1], float(correction[-1])
