"""
Dynamic Simulation Scenarios.

Scenario A: 200 MW step load increase at bus 8.
Scenario B: Generator G6 outage (bus 35, 650 MW).
Scenario C: Solar irradiance variation causing time-varying bus inertia.

Compares system response with and without GFM virtual inertia.
"""

import numpy as np
from scipy.integrate import solve_ivp

from core.frequency_model import (build_state_space, simulate_frequency,
                                   find_frequency_nadir, compute_rocof)


# ─────────────────────────────────────────────────────────────────
# Scenario A: Step load increase
# ─────────────────────────────────────────────────────────────────

def scenario_A_load_step(H_B_sync, H_B_total, load_buses_1idx,
                          delta_P_mw=200.0, S_total_mva=7700.0,
                          t_end=20.0, dt=0.01):
    """
    Scenario A: 200 MW step load increase at bus 8.

    Simulates frequency response at every load bus for two cases:
      (1) synchronous inertia only
      (2) synchronous + GFM virtual inertia

    Parameters
    ----------
    H_B_sync        : np.ndarray (n_load,)
    H_B_total       : np.ndarray (n_load,), includes virtual inertia
    load_buses_1idx : list of int
    delta_P_mw      : float, step size [MW]
    S_total_mva     : float
    t_end           : float
    dt              : float

    Returns
    -------
    results : dict with keys 'time', 'freq_sync', 'freq_total',
                              'rocof_sync', 'rocof_total',
                              'nadir_sync', 'nadir_total'
    """
    delta_P_pu = delta_P_mw / S_total_mva
    D_damp     = 2.0
    n_load     = len(H_B_sync)

    results = {
        "time":        None,
        "freq_sync":   [],
        "freq_total":  [],
        "rocof_sync":  np.zeros(n_load),
        "rocof_total": np.zeros(n_load),
        "nadir_sync":  np.zeros(n_load),
        "nadir_total": np.zeros(n_load),
    }

    for i, (H_s, H_t) in enumerate(zip(H_B_sync, H_B_total)):
        H_s = max(H_s, 0.5)
        H_t = max(H_t, 0.5)

        A_s, B_s, _ = build_state_space(H_s, D_damp)
        t_s, _, f_s, roc_s = simulate_frequency(A_s, B_s, delta_P_pu,
                                                  (0, t_end), dt)

        A_t, B_t, _ = build_state_space(H_t, D_damp)
        t_t, _, f_t, roc_t = simulate_frequency(A_t, B_t, delta_P_pu,
                                                  (0, t_end), dt)

        results["freq_sync"].append(f_s)
        results["freq_total"].append(f_t)
        results["rocof_sync"][i]  = float(np.min(roc_s))
        results["rocof_total"][i] = float(np.min(roc_t))
        nadir_s, _ = find_frequency_nadir(f_s)
        nadir_t, _ = find_frequency_nadir(f_t)
        results["nadir_sync"][i]  = nadir_s
        results["nadir_total"][i] = nadir_t

    results["time"] = t_s
    results["freq_sync"]  = np.array(results["freq_sync"])   # (n_load x n_t)
    results["freq_total"] = np.array(results["freq_total"])

    print(f"  [Scenario A] delta_P={delta_P_mw} MW  "
          f"worst nadir (sync)={results['nadir_sync'].min():.3f} Hz  "
          f"worst nadir (total)={results['nadir_total'].min():.3f} Hz")
    return results


# ─────────────────────────────────────────────────────────────────
# Scenario B: Generator outage
# ─────────────────────────────────────────────────────────────────

def scenario_B_gen_outage(H_B_sync, H_B_total, D_matrix,
                           load_buses_1idx, tripped_gen_idx=5,
                           P_gen_mw=650.0, S_total_mva=7700.0,
                           t_end=20.0, dt=0.01):
    """
    Scenario B: Generator G6 outage (bus 35, ~650 MW).

    Distributes the disturbance to each bus using electrical distance D,
    then simulates individual bus frequency trajectories.

    Parameters
    ----------
    H_B_sync       : np.ndarray (n_load,)
    H_B_total      : np.ndarray (n_load,)
    D_matrix       : np.ndarray (n_gen x n_load), col-normalised distances
    load_buses_1idx: list of int
    tripped_gen_idx: int, column index of tripped generator in D_matrix
    P_gen_mw       : float, tripped generator power [MW]
    S_total_mva    : float
    t_end, dt      : float

    Returns
    -------
    results : dict with time, trajectories for bus 9 and bus 22,
              RoCoF and nadir arrays
    """
    n_load = len(H_B_sync)
    D_damp = 2.0

    # Distribute disturbance: delta_P_i = D[k_trip, i] * P_trip_pu
    P_trip_pu = P_gen_mw / S_total_mva
    if D_matrix is not None and D_matrix.shape[0] > tripped_gen_idx:
        delta_P_bus = D_matrix[tripped_gen_idx, :] * P_trip_pu  # (n_load,)
    else:
        delta_P_bus = np.full(n_load, P_trip_pu / n_load)

    nadir_sync  = np.zeros(n_load)
    nadir_total = np.zeros(n_load)
    rocof_sync  = np.zeros(n_load)
    rocof_total = np.zeros(n_load)

    # Full trajectory for two highlight buses
    bus9_idx  = next((i for i, b in enumerate(load_buses_1idx) if b == 9),  0)
    bus22_idx = next((i for i, b in enumerate(load_buses_1idx) if b == 22), 5)

    traj = {}
    for label, idx in [("bus9", bus9_idx), ("bus22", bus22_idx)]:
        H_s = max(H_B_sync[idx],  0.5)
        H_t = max(H_B_total[idx], 0.5)
        dP  = float(delta_P_bus[idx])

        A_s, B_s, _ = build_state_space(H_s, D_damp)
        t_s, _, f_s, roc_s = simulate_frequency(A_s, B_s, dP, (0, t_end), dt)

        A_t, B_t, _ = build_state_space(H_t, D_damp)
        t_t, _, f_t, roc_t = simulate_frequency(A_t, B_t, dP, (0, t_end), dt)

        traj[f"{label}_sync_f"]  = f_s
        traj[f"{label}_total_f"] = f_t
        traj["time"] = t_s

    for i in range(n_load):
        dP = float(delta_P_bus[i])
        H_s = max(H_B_sync[i],  0.5)
        H_t = max(H_B_total[i], 0.5)
        A_s, B_s, _ = build_state_space(H_s, D_damp)
        _, _, f_s, roc_s = simulate_frequency(A_s, B_s, dP, (0, t_end), dt)
        A_t, B_t, _ = build_state_space(H_t, D_damp)
        _, _, f_t, roc_t = simulate_frequency(A_t, B_t, dP, (0, t_end), dt)
        nadir_sync[i],  _ = find_frequency_nadir(f_s)
        nadir_total[i], _ = find_frequency_nadir(f_t)
        rocof_sync[i]  = float(np.min(roc_s))
        rocof_total[i] = float(np.min(roc_t))

    results = {**traj,
               "nadir_sync": nadir_sync, "nadir_total": nadir_total,
               "rocof_sync": rocof_sync, "rocof_total": rocof_total,
               "delta_P_bus": delta_P_bus}

    print(f"  [Scenario B] G6 trip={P_gen_mw} MW  "
          f"worst nadir (sync)={nadir_sync.min():.3f} Hz  "
          f"worst nadir (total)={nadir_total.min():.3f} Hz")
    return results


# ─────────────────────────────────────────────────────────────────
# Scenario C: Solar irradiance variation
# ─────────────────────────────────────────────────────────────────

def scenario_C_solar_variation(H_B_base, H_B_total_base, load_buses_1idx,
                                 t_end=20.0, dt=0.5):
    """
    Scenario C: Time-varying VRES output causes time-varying bus inertia.

    Models solar irradiance as a sinusoidal + ramp variation over t_end seconds,
    scaling the inertia contribution proportionally.

    Parameters
    ----------
    H_B_base        : np.ndarray, synchronous bus inertia [s]
    H_B_total_base  : np.ndarray, total bus inertia at full VRES [s]
    load_buses_1idx : list of int
    t_end           : float
    dt              : float

    Returns
    -------
    results : dict with time, H_B_sync_t, H_B_total_t, H_sys_t
    """
    t = np.arange(0, t_end + dt, dt)
    n_t    = len(t)
    n_load = len(H_B_base)

    # Solar irradiance profile: ramp up + sinusoidal fluctuation
    irradiance = 0.6 + 0.3 * np.sin(2 * np.pi * t / 10) + 0.1 * (t / t_end)
    irradiance = np.clip(irradiance, 0.0, 1.0)

    H_virt_base = H_B_total_base - H_B_base   # virtual contribution at full VRES

    H_B_sync_t  = np.outer(H_B_base,   np.ones(n_t))          # (n_load x n_t)
    H_B_total_t = H_B_sync_t + np.outer(H_virt_base, irradiance)

    H_sys_sync  = H_B_sync_t.mean(axis=0)
    H_sys_total = H_B_total_t.mean(axis=0)

    print(f"  [Scenario C] H_sys range (sync):  "
          f"{H_sys_sync.min():.2f}–{H_sys_sync.max():.2f} s")
    print(f"  [Scenario C] H_sys range (total): "
          f"{H_sys_total.min():.2f}–{H_sys_total.max():.2f} s")

    return {"time": t, "irradiance": irradiance,
            "H_B_sync_t": H_B_sync_t,
            "H_B_total_t": H_B_total_t,
            "H_sys_sync": H_sys_sync,
            "H_sys_total": H_sys_total}


def run_all_scenarios(H_B_sync, H_B_total, D_matrix, load_buses_1idx,
                       S_total_mva=7700.0):
    """Run A, B, C and return all results."""
    print("\n[Dynamic Simulations]")
    res_A = scenario_A_load_step(H_B_sync, H_B_total, load_buses_1idx,
                                  delta_P_mw=200.0,
                                  S_total_mva=S_total_mva)
    res_B = scenario_B_gen_outage(H_B_sync, H_B_total, D_matrix,
                                   load_buses_1idx, tripped_gen_idx=5,
                                   P_gen_mw=650.0,
                                   S_total_mva=S_total_mva)
    res_C = scenario_C_solar_variation(H_B_sync, H_B_total, load_buses_1idx)

    return {"A": res_A, "B": res_B, "C": res_C}
