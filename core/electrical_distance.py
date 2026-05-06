"""
Electrical Distance Computation via Kron Reduction.

Reference: Ghosh et al., "Assessment of Bus Inertia to Enhance Dynamic
           Flexibility of Hybrid Power Systems With Renewable Energy
           Integration," IEEE Trans. Power Delivery, 2023.  (Eq. 4-7)
"""

import numpy as np


def compute_electrical_distance(Y_bus, gen_buses_0idx, load_buses_0idx):
    """
    Compute the normalised electrical distance matrix D of shape
    (n_generators, n_load_buses) via Kron reduction of Y_bus.

    The weight d_ki represents how much generator k "sees" load bus i,
    i.e. the fraction of generator k's influence felt at bus i.
    Each column sums to 1:  sum_k d_ki = 1  for every load bus i.

    Parameters
    ----------
    Y_bus           : np.ndarray (N x N), complex admittance matrix
    gen_buses_0idx  : list of int, generator bus indices (0-based)
    load_buses_0idx : list of int, load bus indices (0-based)

    Returns
    -------
    D : np.ndarray (n_gen x n_load), float
        D[k, i] = normalised electrical distance from generator k to load bus i
    alpha : np.ndarray (n_gen x n_load), float
        Raw reduced admittance magnitudes before normalisation
    """
    n_gen  = len(gen_buses_0idx)
    n_load = len(load_buses_0idx)

    # --- Kron reduction: eliminate all buses except gen and load buses ------
    # Partition: keep = gen + load, eliminate = everything else
    keep = sorted(set(gen_buses_0idx) | set(load_buses_0idx))
    elim = [b for b in range(Y_bus.shape[0]) if b not in keep]

    Y_red = _kron_reduce(Y_bus, keep, elim)

    # Remap indices into reduced matrix
    keep_arr = np.array(keep)
    gen_red  = [int(np.where(keep_arr == g)[0][0]) for g in gen_buses_0idx]
    load_red = [int(np.where(keep_arr == l)[0][0]) for l in load_buses_0idx]

    # --- alpha_ki = |Z_bus_red[i, k]|  (off-diagonal admittance magnitude) --
    # Use Z_bus of reduced system
    try:
        Z_red = np.linalg.inv(Y_red)
    except np.linalg.LinAlgError:
        Z_red = np.linalg.pinv(Y_red)

    alpha = np.zeros((n_gen, n_load), dtype=float)
    for ki, g in enumerate(gen_red):
        for li, l in enumerate(load_red):
            alpha[ki, li] = abs(Z_red[l, g]) + 1e-12   # avoid division by zero

    # --- Normalise each column so sum over generators = 1 ------------------
    # d_ki = alpha_ki / sum_k(alpha_ki)   (Ghosh eq. 6)
    col_sums = alpha.sum(axis=0, keepdims=True)   # (1 x n_load)
    col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
    D = alpha / col_sums

    return D, alpha


def _kron_reduce(Y, keep_idx, elim_idx):
    """
    Kron (Schur complement) reduction of Y_bus.
    Eliminates the buses in elim_idx, retaining keep_idx.

    Returns Y_reduced of shape (len(keep_idx), len(keep_idx)).
    """
    if not elim_idx:
        idx = np.ix_(keep_idx, keep_idx)
        return Y[idx].copy()

    # Partition Y into [Y_kk, Y_ke; Y_ek, Y_ee]
    Y_kk = Y[np.ix_(keep_idx, keep_idx)]
    Y_ke = Y[np.ix_(keep_idx, elim_idx)]
    Y_ek = Y[np.ix_(elim_idx, keep_idx)]
    Y_ee = Y[np.ix_(elim_idx, elim_idx)]

    try:
        Y_ee_inv = np.linalg.inv(Y_ee)
    except np.linalg.LinAlgError:
        Y_ee_inv = np.linalg.pinv(Y_ee)

    # Schur complement
    Y_red = Y_kk - Y_ke @ Y_ee_inv @ Y_ek
    return Y_red


def compute_electrical_distance_simple(Y_bus, gen_buses_0idx, load_buses_0idx):
    """
    Simpler version using Z_bus directly (no Kron reduction).
    Uses |Z_bus[i,k]| as proximity measure.
    Faster but less accurate for large networks.

    Returns D (n_gen x n_load), normalised column-wise.
    """
    try:
        Z_bus = np.linalg.inv(Y_bus)
    except np.linalg.LinAlgError:
        Z_bus = np.linalg.pinv(Y_bus)

    n_gen  = len(gen_buses_0idx)
    n_load = len(load_buses_0idx)
    alpha  = np.zeros((n_gen, n_load), dtype=float)

    for ki, g in enumerate(gen_buses_0idx):
        for li, l in enumerate(load_buses_0idx):
            alpha[ki, li] = abs(Z_bus[l, g]) + 1e-12

    col_sums = alpha.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
    D = alpha / col_sums
    return D, alpha
