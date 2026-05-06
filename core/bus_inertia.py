"""
Bus Inertia Estimation — Ghosh et al. 2023 Method.

Reference: Ghosh et al., "Assessment of Bus Inertia to Enhance Dynamic
           Flexibility of Hybrid Power Systems With Renewable Energy
           Integration," IEEE Trans. Power Delivery, 2023.
           Equations (8)-(14).
"""

import numpy as np
from core.electrical_distance import compute_electrical_distance_simple


def estimate_bus_inertia(Y_bus, H_generators, gen_buses_0idx, load_buses_0idx,
                          S_gen_mva=None, S_base=100.0):
    """
    Estimate effective bus inertia H_B for all load buses using the
    admittance-weighted method of Ghosh et al. 2023 (Eq. 14).

    H_B[i] = sum_k  W_c[i,k] * H_k * S_k / S_base

    where W_c[i,k] is the row-normalised electrical distance weight:
        W_c[i,k] = alpha[i,k] / sum_j(alpha[i,j])   (rows sum to 1)

    Parameters
    ----------
    Y_bus           : np.ndarray (N x N) complex
    H_generators    : array-like, H constants [s] for each generator
    gen_buses_0idx  : list of int, generator bus indices (0-based)
    load_buses_0idx : list of int, load bus indices (0-based)
    S_gen_mva       : array-like or None, MVA ratings of generators
                      If None, assumes all equal (H_B = weighted average of H)
    S_base          : float, system base MVA (default 100)

    Returns
    -------
    H_B       : np.ndarray (n_load,), effective inertia at each load bus [s]
    W_c       : np.ndarray (n_load, n_gen), row-normalised weight matrix
    D         : np.ndarray (n_gen, n_load), column-normalised distance matrix
    """
    H_generators = np.asarray(H_generators, dtype=float)
    n_gen  = len(gen_buses_0idx)
    n_load = len(load_buses_0idx)

    if S_gen_mva is None:
        S_gen_mva = np.ones(n_gen) * S_base
    else:
        S_gen_mva = np.asarray(S_gen_mva, dtype=float)

    # D: (n_gen x n_load), column-normalised (sum over generators = 1)
    D, alpha = compute_electrical_distance_simple(
        Y_bus, gen_buses_0idx, load_buses_0idx)

    # W_c: (n_load x n_gen), row-normalised (sum over generators = 1 per bus)
    # Transpose alpha so rows = load buses, cols = generators
    alpha_T  = alpha.T   # (n_load x n_gen)
    row_sums = alpha_T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    W_c = alpha_T / row_sums   # (n_load x n_gen)

    # Inertia energy per generator on system base: H_k * S_k / S_base
    H_scaled = H_generators * S_gen_mva / S_base   # (n_gen,)

    # H_B[i] = W_c[i,:] @ H_scaled   (Ghosh eq. 14)
    H_B = W_c @ H_scaled   # (n_load,)

    return H_B, W_c, D


def identify_weak_buses(H_B, load_buses_1idx, threshold_percentile=25):
    """
    Identify inertia-weak buses below a given percentile threshold.

    Parameters
    ----------
    H_B                : np.ndarray, bus inertia values
    load_buses_1idx    : list of int, 1-based bus numbers
    threshold_percentile: float

    Returns
    -------
    weak_buses : list of (bus_number, H_B_value), sorted ascending by H_B
    threshold  : float, the H_B threshold value
    """
    threshold = float(np.percentile(H_B, threshold_percentile))
    weak = [(load_buses_1idx[i], float(H_B[i]))
            for i in range(len(H_B)) if H_B[i] <= threshold]
    weak.sort(key=lambda x: x[1])
    return weak, threshold


def compute_inertia_distribution_summary(H_B, load_buses_1idx):
    """Return a dict of summary statistics for the bus inertia distribution."""
    return {
        "min":    float(H_B.min()),
        "max":    float(H_B.max()),
        "mean":   float(H_B.mean()),
        "std":    float(H_B.std()),
        "p25":    float(np.percentile(H_B, 25)),
        "p75":    float(np.percentile(H_B, 75)),
        "weakest_bus":   load_buses_1idx[int(H_B.argmin())],
        "strongest_bus": load_buses_1idx[int(H_B.argmax())],
    }
