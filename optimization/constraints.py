"""
Inertia and Power System Constraint Functions.

Implements minimum inertia thresholds based on RoCoF grid code limit,
power balance checks, and voltage limit checks.
"""

import numpy as np


def compute_min_inertia_thresholds(D_matrix, delta_P_mw_per_gen,
                                    S_total_mva, rocof_limit=1.0, f0=60.0):
    """
    Compute the minimum bus inertia H_min[i] required to keep RoCoF
    within the grid-code limit at each load bus.

    From swing equation:  RoCoF_i = delta_P_i * f0 / (2 * H_i)
    Rearranged:           H_min_i = delta_P_i * f0 / (2 * rocof_limit)

    where delta_P_i = sum_k  D[k,i] * P_gen_k_pu
    (electrical distance distributes the disturbance to each bus)

    Parameters
    ----------
    D_matrix          : np.ndarray (n_gen x n_load), column-normalised distances
    delta_P_mw_per_gen: array-like (n_gen,), power loss from each generator [MW]
    S_total_mva       : float, system base [MVA]
    rocof_limit       : float, RoCoF limit [Hz/s], default 1.0
    f0                : float, nominal frequency [Hz]

    Returns
    -------
    H_min    : np.ndarray (n_load,), minimum inertia [s] per load bus
    delta_P  : np.ndarray (n_load,), distributed disturbance at each bus [pu]
    """
    delta_P_pu = np.asarray(delta_P_mw_per_gen, dtype=float) / S_total_mva
    # Distribute disturbance to each load bus using D (n_gen x n_load)
    delta_P_bus = D_matrix.T @ delta_P_pu    # (n_load,)
    H_min = delta_P_bus * f0 / (2.0 * rocof_limit)
    H_min = np.maximum(H_min, 0.01)          # floor at 0.01 s
    return H_min, delta_P_bus


def check_inertia_constraint(H_B_total, H_min):
    """
    Check whether each bus satisfies the minimum inertia constraint.

    Parameters
    ----------
    H_B_total : np.ndarray, total bus inertia (sync + virtual) [s]
    H_min     : np.ndarray, minimum required inertia [s]

    Returns
    -------
    satisfied  : np.ndarray (bool), True where constraint is met
    deficit    : np.ndarray (float), H_min - H_B_total (>0 = deficit)
    n_violated : int, number of buses violating constraint
    """
    H_B_total = np.asarray(H_B_total, dtype=float)
    H_min     = np.asarray(H_min,     dtype=float)
    deficit   = H_min - H_B_total
    satisfied = deficit <= 0
    return satisfied, deficit, int((~satisfied).sum())


def check_voltage_limits(V_pu, V_min=0.95, V_max=1.05):
    """
    Check nodal voltage magnitude limits.

    Parameters
    ----------
    V_pu  : array-like, bus voltages in pu
    V_min : float
    V_max : float

    Returns
    -------
    ok    : np.ndarray (bool)
    n_viol: int
    """
    V = np.asarray(V_pu, dtype=float)
    ok = (V >= V_min) & (V <= V_max)
    return ok, int((~ok).sum())


def check_power_balance(P_gen_mw, P_load_mw, P_vres_mw, tolerance_mw=1.0):
    """
    Simple active power balance check.

    Parameters
    ----------
    P_gen_mw  : array-like, synchronous generator dispatch [MW]
    P_load_mw : float, total system load [MW]
    P_vres_mw : array-like, VRES dispatch [MW]
    tolerance_mw : float, allowed mismatch [MW]

    Returns
    -------
    balanced : bool
    mismatch : float [MW]
    """
    total_gen  = np.sum(P_gen_mw) + np.sum(P_vres_mw)
    mismatch   = abs(total_gen - P_load_mw)
    return mismatch <= tolerance_mw, float(mismatch)


def compute_transmission_losses(branch_data, V_pu, theta_rad):
    """
    Estimate transmission losses using DC approximation.

    P_loss_ij = G_ij * (Vi^2 + Vj^2 - 2*Vi*Vj*cos(theta_i - theta_j))

    Parameters
    ----------
    branch_data : list of [from, to, r, x, b, rate]
    V_pu        : np.ndarray (N,), bus voltages [pu]
    theta_rad   : np.ndarray (N,), bus angles [rad]

    Returns
    -------
    total_loss_mw : float (on S_base=100 MVA)
    """
    S_BASE = 100.0
    total_loss = 0.0
    for branch in branch_data:
        i = int(branch[0]) - 1
        j = int(branch[1]) - 1
        r = branch[2]
        x = branch[3]
        z2 = r**2 + x**2
        if z2 < 1e-12:
            continue
        G_ij = r / z2
        Vi, Vj = V_pu[i], V_pu[j]
        dtheta = theta_rad[i] - theta_rad[j]
        loss = G_ij * (Vi**2 + Vj**2 - 2 * Vi * Vj * np.cos(dtheta))
        total_loss += loss
    return total_loss * S_BASE   # convert to MW
