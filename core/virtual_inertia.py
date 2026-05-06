"""
Grid-Forming (GFM) Inverter Virtual Inertia Contribution.

Novel Contribution: Extends Ghosh 2023 bus inertia framework to include
virtual inertia from Grid-Forming inverters modelled as Virtual Synchronous
Generators (VSG).

GFM swing equation (VSG model):
    2 * H_virt * d(omega)/dt = P_ref - P_e - D_virt * (omega - omega_0)

The virtual inertia contribution at each load bus is mapped using the
same electrical distance weighting as the synchronous generators.
"""

import numpy as np
from core.electrical_distance import compute_electrical_distance_simple


def compute_virtual_inertia_contribution(Y_bus, gfm_buses_0idx, H_virt_list,
                                          S_virt_mva, load_buses_0idx,
                                          S_base=100.0):
    """
    Compute the virtual inertia contribution H_B_virt at each load bus
    from GFM inverters, using electrical distance weighting.

    H_B_virt[i] = sum_k  W_gfm[i,k] * H_virt_k * S_virt_k / S_base

    Parameters
    ----------
    Y_bus           : np.ndarray (N x N), complex admittance matrix
    gfm_buses_0idx  : list of int, GFM inverter bus indices (0-based)
    H_virt_list     : array-like, virtual inertia constants [s] for each GFM
    S_virt_mva      : array-like, MVA ratings of GFM inverters
    load_buses_0idx : list of int, load bus indices (0-based)
    S_base          : float, system base MVA

    Returns
    -------
    H_B_virt : np.ndarray (n_load,), virtual inertia at each load bus [s]
    W_gfm    : np.ndarray (n_load x n_gfm), weight matrix
    """
    H_virt   = np.asarray(H_virt_list, dtype=float)
    S_virt   = np.asarray(S_virt_mva,  dtype=float)
    n_load   = len(load_buses_0idx)
    n_gfm    = len(gfm_buses_0idx)

    if n_gfm == 0:
        return np.zeros(n_load), np.zeros((n_load, 0))

    # Electrical distance from GFM buses to load buses
    D_gfm, alpha_gfm = compute_electrical_distance_simple(
        Y_bus, gfm_buses_0idx, load_buses_0idx)

    # Row-normalise (per load bus, weights over GFM units sum to 1)
    alpha_T  = alpha_gfm.T          # (n_load x n_gfm)
    row_sums = alpha_T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    W_gfm    = alpha_T / row_sums   # (n_load x n_gfm)

    H_virt_scaled = H_virt * S_virt / S_base   # (n_gfm,)
    H_B_virt = W_gfm @ H_virt_scaled           # (n_load,)

    return H_B_virt, W_gfm


def combine_total_bus_inertia(H_B_sync, H_B_virt):
    """
    Combine synchronous and virtual inertia at each bus.

    H_B_total[i] = H_B_sync[i] + H_B_virt[i]

    Parameters
    ----------
    H_B_sync : np.ndarray, synchronous generator contribution
    H_B_virt : np.ndarray, GFM virtual inertia contribution

    Returns
    -------
    H_B_total : np.ndarray
    """
    return np.asarray(H_B_sync) + np.asarray(H_B_virt)


def gfm_frequency_response(H_virt, D_virt, P_ref, P_e_init, omega_0,
                             t_span, dt=0.01):
    """
    Simulate GFM inverter frequency response using VSG swing equation.

    2*H_virt * d(omega)/dt = P_ref - P_e - D_virt*(omega - omega_0)

    Assumes P_e steps from P_e_init to P_ref + delta_P at t=1s.

    Parameters
    ----------
    H_virt   : float, virtual inertia [s]
    D_virt   : float, virtual damping [pu]
    P_ref    : float, active power reference [pu]
    P_e_init : float, initial electrical power [pu]
    omega_0  : float, nominal frequency [rad/s]
    t_span   : tuple (t_start, t_end)
    dt       : float, time step [s]

    Returns
    -------
    t      : np.ndarray, time vector
    omega  : np.ndarray, angular frequency [rad/s]
    f      : np.ndarray, frequency [Hz]
    """
    t = np.arange(t_span[0], t_span[1], dt)
    omega = np.zeros(len(t))
    omega[0] = omega_0
    f0_hz = omega_0 / (2 * np.pi)

    t_fault = 1.0  # disturbance at t=1s

    for k in range(1, len(t)):
        P_e = P_e_init if t[k] < t_fault else P_ref + 0.1  # 0.1 pu step
        d_omega = (P_ref - P_e - D_virt * (omega[k-1] - omega_0)) / (2 * H_virt)
        omega[k] = omega[k-1] + d_omega * dt

    f = omega / (2 * np.pi)
    return t, omega, f
