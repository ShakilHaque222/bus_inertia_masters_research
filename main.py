"""
Main Runner — Bus Inertia Research Project
==========================================
Real-Time Dynamic Bus Inertia Estimation Integrating Virtual Inertia from
Grid-Forming Inverters for Optimal VRES Scheduling

Author : Shakil Haque
System : IEEE 39-Bus New England
Date   : 2026

Steps
-----
1. Load IEEE 39-bus system data
2. Compute electrical distances (sync generators + GFM inverters)
3. Estimate bus inertia (Ghosh 2023 method)
4. Add virtual inertia from GFM inverters (Novel contribution)
5. Compute minimum inertia thresholds (Swetala 2025 method)
6. Run NSGA-II multi-objective optimisation
7. Run dynamic simulations (Scenarios A, B, C)
8. Generate and save all 6 publication plots
9. Print summary table
"""

import sys, os, time
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.ieee39_data import (GENERATOR_DATA, GEN_BUSES, LOAD_BUSES,
                               BRANCH_DATA, VRES_DATA, GFM_DATA,
                               build_ybus, get_system_summary)

from core.electrical_distance import compute_electrical_distance_simple
from core.bus_inertia import (estimate_bus_inertia, identify_weak_buses,
                               compute_inertia_distribution_summary)
from core.virtual_inertia import (compute_virtual_inertia_contribution,
                                   combine_total_bus_inertia)
from core.frequency_model import compute_area_rocof

from optimization.constraints import (compute_min_inertia_thresholds,
                                       check_inertia_constraint)
from optimization.nsga2_scheduler import run_optimization

from simulation.dynamic_sim import run_all_scenarios
from results.plots import generate_all_plots


def divider(title=""):
    w = 62
    print("\n" + "=" * w)
    if title:
        pad = (w - len(title) - 2) // 2
        print(" " * pad + f" {title} ")
        print("=" * w)


def main():
    t0 = time.time()
    print("\n" + "=" * 62)
    print("  Bus Inertia Research — IEEE 39-Bus New England System")
    print("  Author: Shakil Haque | University of Fukui | 2026")
    print("=" * 62)

    os.makedirs("results", exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Load system data
    # ══════════════════════════════════════════════════════════════
    divider("STEP 1: System Data")
    H_sys, E_kin = get_system_summary()

    Y_bus   = build_ybus(39)
    gen_0   = [b - 1 for b in GEN_BUSES]     # 0-indexed
    load_0  = [b - 1 for b in LOAD_BUSES]
    gfm_0   = [b - 1 for b in GFM_DATA.keys()]

    H_G     = np.array([GENERATOR_DATA[b]["H"]     for b in GEN_BUSES])
    S_G     = np.array([GENERATOR_DATA[b]["S_mva"] for b in GEN_BUSES])
    S_total = float(S_G.sum())

    H_virt  = np.array([GFM_DATA[b]["H_virt"] for b in GFM_DATA])
    S_virt  = np.array([GFM_DATA[b]["S_mva"]  for b in GFM_DATA])
    P_gen   = np.array([GENERATOR_DATA[b]["P_mw"] for b in GEN_BUSES])

    print(f"  Y_bus shape  : {Y_bus.shape}")
    print(f"  Gen buses    : {GEN_BUSES}")
    print(f"  Load buses   : {len(LOAD_BUSES)} buses (1–29)")
    print(f"  GFM units    : {list(GFM_DATA.keys())}")

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Electrical distances
    # ══════════════════════════════════════════════════════════════
    divider("STEP 2: Electrical Distances")
    D_sync, alpha_sync = compute_electrical_distance_simple(
        Y_bus, gen_0, load_0)
    D_gfm,  alpha_gfm  = compute_electrical_distance_simple(
        Y_bus, gfm_0, load_0)
    print(f"  D_sync shape : {D_sync.shape}  (n_gen x n_load)")
    print(f"  D_gfm  shape : {D_gfm.shape}   (n_gfm x n_load)")
    print(f"  Column sums D_sync: "
          f"min={D_sync.sum(0).min():.4f}  max={D_sync.sum(0).max():.4f}")

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Bus inertia — Ghosh 2023
    # ══════════════════════════════════════════════════════════════
    divider("STEP 3: Bus Inertia (Ghosh 2023)")
    H_B_sync, W_c, _ = estimate_bus_inertia(
        Y_bus, H_G, gen_0, load_0, S_G)

    stats = compute_inertia_distribution_summary(H_B_sync, LOAD_BUSES)
    print(f"  H_B_sync  min  : {stats['min']:.2f} s  (Bus {stats['weakest_bus']})")
    print(f"  H_B_sync  max  : {stats['max']:.2f} s  (Bus {stats['strongest_bus']})")
    print(f"  H_B_sync  mean : {stats['mean']:.2f} s")
    print(f"  H_B_sync  std  : {stats['std']:.2f} s")

    weak, threshold = identify_weak_buses(H_B_sync, LOAD_BUSES, 25)
    print(f"\n  Weak buses (bottom 25%, H_B < {threshold:.2f} s): "
          f"{[b for b, _ in weak]}")

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Virtual inertia — Novel contribution
    # ══════════════════════════════════════════════════════════════
    divider("STEP 4: GFM Virtual Inertia (Novel Contribution)")
    H_B_virt, W_gfm = compute_virtual_inertia_contribution(
        Y_bus, gfm_0, H_virt, S_virt, load_0)
    H_B_total = combine_total_bus_inertia(H_B_sync, H_B_virt)

    print(f"  GFM inverters : Bus {list(GFM_DATA.keys())}, "
          f"H_virt = {list(H_virt)}")
    print(f"  H_B_virt  min : {H_B_virt.min():.3f} s")
    print(f"  H_B_virt  max : {H_B_virt.max():.3f} s")
    print(f"  H_B_total min : {H_B_total.min():.2f} s")
    print(f"  H_B_total max : {H_B_total.max():.2f} s")
    improvement = (H_B_total - H_B_sync) / H_B_sync * 100
    print(f"  Improvement   : {improvement.min():.1f}–{improvement.max():.1f}%")

    # ══════════════════════════════════════════════════════════════
    # STEP 5: Minimum inertia thresholds
    # ══════════════════════════════════════════════════════════════
    divider("STEP 5: Minimum Inertia Thresholds")
    H_min, delta_P_bus = compute_min_inertia_thresholds(
        D_sync, P_gen, S_total_mva=S_total, rocof_limit=1.0)

    sat_sync,  deficit_sync,  n_viol_sync  = check_inertia_constraint(
        H_B_sync,  H_min)
    sat_total, deficit_total, n_viol_total = check_inertia_constraint(
        H_B_total, H_min)

    print(f"  H_min range   : {H_min.min():.2f}–{H_min.max():.2f} s")
    print(f"  Violations (sync only)  : {n_viol_sync}  / {len(LOAD_BUSES)}")
    print(f"  Violations (with GFM)   : {n_viol_total} / {len(LOAD_BUSES)}")
    gfm_fix = n_viol_sync - n_viol_total
    print(f"  Buses fixed by GFM      : {gfm_fix}")

    # ══════════════════════════════════════════════════════════════
    # STEP 6: NSGA-II optimisation
    # ══════════════════════════════════════════════════════════════
    divider("STEP 6: NSGA-II Optimisation")
    system_data = {
        "vres":       VRES_DATA,
        "generators": GENERATOR_DATA,
        "loads":      {},
    }
    _, pareto_F, pareto_X = run_optimization(
        system_data, H_B_total, H_min, BRANCH_DATA,
        pop_size=50, n_gen=80)

    if pareto_F is not None and len(pareto_F) > 0:
        best_pen_idx  = int(np.argmax(pareto_F[:, 0]))
        best_loss_idx = int(np.argmin(pareto_F[:, 1]))
        print(f"  Best penetration : {pareto_F[best_pen_idx,  0]:.1f}%  "
              f"(losses = {pareto_F[best_pen_idx, 1]:.0f} MW-equiv)")
        print(f"  Best losses      : {pareto_F[best_loss_idx, 1]:.0f} MW-equiv  "
              f"(penetration = {pareto_F[best_loss_idx, 0]:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # STEP 7: Dynamic simulations
    # ══════════════════════════════════════════════════════════════
    divider("STEP 7: Dynamic Simulations")
    sim_results = run_all_scenarios(
        H_B_sync, H_B_total, D_sync, LOAD_BUSES, S_total_mva=S_total)

    # ══════════════════════════════════════════════════════════════
    # STEP 8: Generate plots
    # ══════════════════════════════════════════════════════════════
    divider("STEP 8: Generating Plots")
    plot_paths = generate_all_plots(
        H_B_sync, H_B_total, H_min,
        LOAD_BUSES, pareto_F, sim_results)

    # ══════════════════════════════════════════════════════════════
    # STEP 9: Summary table
    # ══════════════════════════════════════════════════════════════
    divider("STEP 9: Summary")

    print(f"\n  {'Bus':>4}  {'H_sync':>8}  {'H_virt':>8}  "
          f"{'H_total':>8}  {'H_min':>7}  {'Deficit':>8}  {'OK?':>5}")
    print("  " + "-" * 57)
    for i, bus in enumerate(LOAD_BUSES):
        deficit = H_min[i] - H_B_total[i]
        ok = "YES" if deficit <= 0 else "NO"
        print(f"  {bus:>4}  {H_B_sync[i]:>8.2f}  {H_B_virt[i]:>8.3f}  "
              f"{H_B_total[i]:>8.2f}  {H_min[i]:>7.2f}  "
              f"{max(deficit,0):>8.3f}  {ok:>5}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"  Analysis complete in {elapsed:.1f} s")
    print(f"  Results saved to:  results/")
    print(f"  Plots generated:   {len(plot_paths)}")
    print(f"{'=' * 62}\n")

    return {
        "H_B_sync":  H_B_sync,
        "H_B_virt":  H_B_virt,
        "H_B_total": H_B_total,
        "H_min":     H_min,
        "pareto_F":  pareto_F,
        "sim":       sim_results,
    }


if __name__ == "__main__":
    main()
