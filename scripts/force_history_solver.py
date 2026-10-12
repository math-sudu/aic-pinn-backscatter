"""Incremental force equilibrium with the same history used in PINN residuals."""

import numpy as np
from scipy.sparse.linalg import minres


def solve_history_step(problem, force_n, previous_states, initial, preconditioner):
    """Solve one physical step without advancing history during Newton updates.

    The complete converged prefix enters every residual evaluation. Previous
    states are fixed only for this sequential numerical solve; the differentiable
    path residual itself continues to couple all steps during neural training.
    The currently configured tractions apply to the returned current residual.
    """
    p = problem.p
    prefix = np.asarray(previous_states).reshape(-1, p.ndof)
    if len(prefix):
        prior_history = p.forces(p.tensor(prefix)[None])[2][0, -1].numpy()
    else:
        prior_history = np.zeros(len(p.mesh["interface_area"])*len(p.surface_weights))

    def evaluate(current):
        path = np.vstack([prefix, current])
        # Current external forces broadcast over the prefix; only the current
        # residual is solved here, while all prefix separations build history.
        residual, damage, _ = problem.equilibrium(p.tensor(path)[None], p.tensor([[force_n]]))
        return residual[0, -1].numpy(), float(damage[0, -1].max())

    current = np.array(initial, copy=True)
    residual, damage = evaluate(current)
    for iteration in range(80):
        if np.max(np.abs(residual)) < 1e-5:
            return current, iteration, damage
        tangent = problem.history_tangent(current, prior_history)
        # Cohesive softening can have an indefinite algorithmic tangent.
        step, status = minres(tangent, -residual, M=preconditioner, rtol=1e-12)
        if status != 0:
            raise RuntimeError(f"History tangent solve failed: {status}")
        scale = 1.
        norm = np.linalg.norm(residual)
        while scale >= 1e-10:
            candidate = current+scale*step
            trial, trial_damage = evaluate(candidate)
            if np.linalg.norm(trial) < (1-1e-4*scale)*norm or np.max(np.abs(trial)) < 1e-5:
                current, residual, damage = candidate, trial, trial_damage
                break
            scale *= .5
        else:
            raise RuntimeError("No force-residual decrease along the history Newton step")
    raise RuntimeError("Force-history equilibrium did not converge")
