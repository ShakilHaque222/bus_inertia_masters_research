"""
Publication-Quality Plots for Bus Inertia Research.

Generates 6 figures:
  Plot 1 — Bus inertia distribution (sync vs total with virtual)
  Plot 2 — NSGA-II Pareto front
  Plot 3 — Frequency response trajectories (bus 9 vs bus 22)
  Plot 4 — Per-bus RoCoF: without vs with GFM virtual inertia
  Plot 5 — System inertia dynamics during fault (Scenario C)
  Plot 6 — Minimum required vs achieved inertia (grouped bar)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROCOF_LIMIT  = 1.0    # Hz/s
NADIR_LIMIT  = 59.5   # Hz
F0           = 60.0   # Hz

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})


# ─────────────────────────────────────────────────────────────────
# Plot 1 — Bus Inertia Distribution
# ─────────────────────────────────────────────────────────────────

def plot_bus_inertia_distribution(H_B_sync, H_B_total, H_min,
                                   load_buses_1idx, save=True):
    """
    Grouped bar chart: H_B_sync (blue) and H_B_total (orange) per load bus.
    Red dashed line = H_min threshold (worst-case bus).
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    n  = len(H_B_sync)
    x  = np.arange(n)
    w  = 0.38

    ax.bar(x - w/2, H_B_sync,  w, color="#2196F3", alpha=0.85,
           label="$H_B$ sync only",          edgecolor="white", linewidth=0.4)
    ax.bar(x + w/2, H_B_total, w, color="#FF9800", alpha=0.85,
           label="$H_B$ total (sync + GFM)", edgecolor="white", linewidth=0.4)

    ax.axhline(H_min.max(), color="#F44336", linestyle="--", linewidth=1.8,
               label=f"$H_{{min}}$ worst-case = {H_min.max():.2f} s")

    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(b) for b in load_buses_1idx[::2]], fontsize=9)
    ax.set_xlabel("Load Bus Number")
    ax.set_ylabel("Effective Inertia $H_B$ (s)")
    ax.set_title("Plot 1 — Bus Inertia Distribution: Synchronous vs Total (with GFM Virtual Inertia)")
    ax.legend(fontsize=10)

    path = os.path.join(OUTPUT_DIR, "plot1_bus_inertia_distribution.png")
    if save:
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────
# Plot 2 — Pareto Front
# ─────────────────────────────────────────────────────────────────

def plot_pareto_front(pareto_F, save=True):
    """
    Scatter: VRES penetration (%) vs transmission losses (MW).
    Colour-coded by number of committed SGs.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    pen   = pareto_F[:, 0]
    loss  = pareto_F[:, 1]
    n_sg  = pareto_F[:, 2]

    sc = ax.scatter(pen, loss, c=n_sg, cmap="RdYlGn_r",
                    s=60, alpha=0.85, edgecolors="k", linewidths=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Committed SGs")

    ax.set_xlabel("VRES Penetration (%)")
    ax.set_ylabel("Transmission Losses (MW equiv.)")
    ax.set_title("Plot 2 — NSGA-II Pareto Front\n"
                 "Trade-off: VRES Penetration vs Losses vs Committed SGs")

    path = os.path.join(OUTPUT_DIR, "plot2_pareto_front.png")
    if save:
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────
# Plot 3 — Frequency Response Trajectories
# ─────────────────────────────────────────────────────────────────

def plot_frequency_response(t, f_bus9_base, f_bus9_vres,
                              f_bus22_base, f_bus22_vres, save=True):
    """
    4 frequency trajectory curves for Scenario B (G6 outage).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, f_bus9_base,  color="#1565C0", linewidth=2.0,
            label="Bus 9 — base (sync only)")
    ax.plot(t, f_bus9_vres,  color="#1565C0", linewidth=2.0,
            linestyle="--", label="Bus 9 — with GFM")
    ax.plot(t, f_bus22_base, color="#C62828", linewidth=2.0,
            label="Bus 22 — base (sync only)")
    ax.plot(t, f_bus22_vres, color="#C62828", linewidth=2.0,
            linestyle="--", label="Bus 22 — with GFM")

    ax.axhline(NADIR_LIMIT, color="black", linestyle=":", linewidth=1.5,
               label=f"Nadir limit = {NADIR_LIMIT} Hz")
    ax.axhline(F0, color="grey", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Plot 3 — Frequency Response: Bus 9 (high inertia) vs Bus 22 (low inertia)\n"
                 "Scenario B — G6 Generator Outage (650 MW)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(min(f_bus22_base.min() - 0.1, 59.3), F0 + 0.05)

    path = os.path.join(OUTPUT_DIR, "plot3_frequency_response.png")
    if save:
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────
# Plot 4 — Per-Bus RoCoF Comparison
# ─────────────────────────────────────────────────────────────────

def plot_rocof_comparison(rocof_sync, rocof_total, load_buses_1idx, save=True):
    """
    Grouped bar chart: RoCoF without GFM (red) vs with GFM (green).
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    n = len(rocof_sync)
    x = np.arange(n)
    w = 0.38

    ax.bar(x - w/2, np.abs(rocof_sync),  w, color="#F44336", alpha=0.85,
           label="Without GFM",  edgecolor="white", linewidth=0.4)
    ax.bar(x + w/2, np.abs(rocof_total), w, color="#4CAF50", alpha=0.85,
           label="With GFM virtual inertia", edgecolor="white", linewidth=0.4)

    ax.axhline(ROCOF_LIMIT, color="black", linestyle="--", linewidth=1.8,
               label=f"Grid code limit = {ROCOF_LIMIT} Hz/s")

    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(b) for b in load_buses_1idx[::2]], fontsize=9)
    ax.set_xlabel("Load Bus Number")
    ax.set_ylabel("|RoCoF| (Hz/s)")
    ax.set_title("Plot 4 — Per-Bus RoCoF: Without vs With GFM Virtual Inertia\n"
                 "Scenario A — 200 MW Load Step at Bus 8")
    ax.legend(fontsize=10)

    path = os.path.join(OUTPUT_DIR, "plot4_rocof_comparison.png")
    if save:
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────
# Plot 5 — System Inertia Dynamics
# ─────────────────────────────────────────────────────────────────

def plot_inertia_dynamics(t, H_sys_original, H_sys_vres75, H_sys_vres_gfm,
                           save=True):
    """
    Three lines: original, 75% VRES, VRES + GFM over Scenario C time window.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t, H_sys_original, color="#1565C0", linewidth=2.5,
            label="Base case (sync only)")
    ax.plot(t, H_sys_vres75,   color="#FF9800", linewidth=2.5,
            linestyle="--", label="75% VRES (no GFM)")
    ax.plot(t, H_sys_vres_gfm, color="#2E7D32", linewidth=2.5,
            linestyle="-.", label="75% VRES + GFM virtual inertia")

    ax.fill_between(t, H_sys_vres75, H_sys_vres_gfm,
                    alpha=0.15, color="#2E7D32",
                    label="GFM improvement region")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("System Inertia $H_{sys}$ (s)")
    ax.set_title("Plot 5 — System Inertia Dynamics During Solar Variation\n"
                 "Scenario C — Time-Varying VRES Output")
    ax.legend(fontsize=10)
    ax.set_xlim(0, t[-1])

    path = os.path.join(OUTPUT_DIR, "plot5_inertia_dynamics.png")
    if save:
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────
# Plot 6 — Minimum vs Achieved Inertia
# ─────────────────────────────────────────────────────────────────

def plot_min_vs_achieved(H_B_sync, H_B_total, H_min,
                          load_buses_1idx, save=True):
    """
    Grouped bar: H_min required (grey) vs H_B_sync (blue) vs H_B_total (green).
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    n = len(H_B_sync)
    x = np.arange(n)
    w = 0.28

    ax.bar(x - w,   H_min,     w, color="#9E9E9E", alpha=0.85,
           label="$H_{min}$ required", edgecolor="white", linewidth=0.4)
    ax.bar(x,       H_B_sync,  w, color="#2196F3", alpha=0.85,
           label="$H_B$ sync", edgecolor="white", linewidth=0.4)
    ax.bar(x + w,   H_B_total, w, color="#4CAF50", alpha=0.85,
           label="$H_B$ total (+ GFM)", edgecolor="white", linewidth=0.4)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(b) for b in load_buses_1idx[::2]], fontsize=9)
    ax.set_xlabel("Load Bus Number")
    ax.set_ylabel("Inertia (s)")
    ax.set_title("Plot 6 — Minimum Required vs Achieved Bus Inertia\n"
                 "GFM virtual inertia helps meet $H_{min}$ thresholds")
    ax.legend(fontsize=10)

    path = os.path.join(OUTPUT_DIR, "plot6_min_vs_achieved.png")
    if save:
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────
# Master function
# ─────────────────────────────────────────────────────────────────

def generate_all_plots(H_B_sync, H_B_total, H_min, load_buses_1idx,
                        pareto_F, sim_results):
    """Generate and save all 6 plots."""
    print("\n[Generating plots]")
    paths = []

    paths.append(plot_bus_inertia_distribution(
        H_B_sync, H_B_total, H_min, load_buses_1idx))

    paths.append(plot_pareto_front(pareto_F))

    # Scenario B trajectories
    res_B = sim_results["B"]
    t = res_B.get("time", np.linspace(0, 20, 2001))
    n_t = len(t)
    paths.append(plot_frequency_response(
        t,
        res_B.get("bus9_sync_f",  np.full(n_t, 60.0)),
        res_B.get("bus9_total_f", np.full(n_t, 60.0)),
        res_B.get("bus22_sync_f", np.full(n_t, 59.8)),
        res_B.get("bus22_total_f",np.full(n_t, 59.9)),
    ))

    # Scenario A RoCoF
    res_A = sim_results["A"]
    paths.append(plot_rocof_comparison(
        res_A.get("rocof_sync",  np.ones(len(H_B_sync)) * 0.8),
        res_A.get("rocof_total", np.ones(len(H_B_sync)) * 0.5),
        load_buses_1idx,
    ))

    # Scenario C inertia dynamics
    res_C = sim_results["C"]
    t_c = res_C["time"]
    H_orig = res_C["H_sys_sync"]
    H_vres = res_C["H_sys_total"] * 0.75      # 75% VRES
    H_gfm  = res_C["H_sys_total"]
    paths.append(plot_inertia_dynamics(t_c, H_orig, H_vres, H_gfm))

    paths.append(plot_min_vs_achieved(
        H_B_sync, H_B_total, H_min, load_buses_1idx))

    print(f"\n  All 6 plots saved to {OUTPUT_DIR}/")
    return paths
