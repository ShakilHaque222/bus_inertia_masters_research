# Bus Inertia Research — Master's Thesis

## Research Title
**Real-Time Dynamic Bus Inertia Estimation Integrating Virtual Inertia from Grid-Forming Inverters for Optimal VRES Scheduling**

**Author:** Shakil Haque
**Institution:** Department of Electrical and Electronic Engineering, University of Fukui, Japan
**Year:** 2026

---

## Motivation

Modern power grids are undergoing rapid decarbonisation through the integration of Variable Renewable Energy Sources (VRES) such as wind and solar photovoltaics. While VRES reduce carbon emissions, they displace synchronous generators — the primary source of rotational inertia in conventional grids.

**The problem:** Reduced inertia causes faster frequency deviations (high Rate of Change of Frequency, RoCoF) after disturbances, risking cascading outages and blackouts.

**Gap in Paper 1 (Ghosh et al. 2023):**
- Ghosh computes spatially-resolved bus inertia H_B using the admittance matrix, but:
  - Does **not** compute per-bus RoCoF
  - Does **not** include virtual inertia from Grid-Forming (GFM) inverters
  - Does **not** provide VSG sizing or placement optimisation

**Gap in Paper 2 (Swetala et al. 2025):**
- Swetala optimises VRES scheduling with frequency constraints, but:
  - Uses system-average inertia (loses spatial resolution)
  - Does not map GFM virtual inertia to individual buses

---

## Novel Contribution

This work bridges both gaps by:

1. **Extending Ghosh 2023** to include GFM inverter virtual inertia at the bus level:

$$H_{B,total}^{(i)} = H_{B,sync}^{(i)} + H_{B,virt}^{(i)}$$

where:

$$H_{B,virt}^{(i)} = \sum_k W_{gfm}^{(i,k)} \cdot H_{virt,k} \cdot \frac{S_{virt,k}}{S_{base}}$$

2. **Bus-specific RoCoF** from the swing equation:

$$\text{RoCoF}_i = \frac{-\Delta P_i \cdot f_0}{2 \cdot H_{B,total}^{(i)}}$$

3. **NSGA-II multi-objective optimisation** (extending Swetala 2025) with spatially-resolved inertia constraints:

$$\min \left[ -\text{VRES penetration},\; P_{loss},\; N_{SG} \right]$$

subject to: $H_{B,total}^{(i)} \geq H_{min}^{(i)},\; \forall i$

---

## Methodology

```
IEEE 39-Bus Data
      │
      ▼
Y_bus (39×39 admittance matrix)
      │
      ├──► Electrical Distance D_sync (n_gen × n_load)   [Ghosh eq. 4-7]
      │         computed via Z_bus off-diagonal elements
      │
      ├──► Bus Inertia H_B_sync (29 load buses)           [Ghosh eq. 14]
      │         H_B[i] = Σ_k W_c[i,k] · H_k · S_k/S_base
      │
      ├──► GFM Virtual Inertia H_B_virt                   [Novel]
      │         same electrical distance framework, GFM buses only
      │
      ├──► H_B_total = H_B_sync + H_B_virt                [Novel]
      │
      ├──► H_min thresholds (RoCoF ≤ 1.0 Hz/s)           [Swetala eq. A1]
      │
      ├──► NSGA-II Optimisation (3-objective Pareto front) [Swetala §III]
      │         Decision vars: VRES dispatch + SG commitment
      │
      └──► Dynamic Simulations (Scenarios A, B, C)
                └──► 6 Publication Plots
```

---

## File Structure

```
bus_inertia_masters_research/
├── data/
│   └── ieee39_data.py        — Bus, branch, generator, VRES, GFM data
├── core/
│   ├── electrical_distance.py — Kron reduction + normalised distance D
│   ├── bus_inertia.py         — Ghosh 2023 H_B estimation
│   ├── virtual_inertia.py     — GFM VSG model + H_B_virt computation
│   └── frequency_model.py     — Swing equation, state-space, RoCoF, nadir
├── optimization/
│   ├── constraints.py         — H_min, voltage, power balance checks
│   └── nsga2_scheduler.py     — pymoo NSGA-II 3-objective scheduler
├── simulation/
│   └── dynamic_sim.py         — Scenarios A (load step), B (gen outage), C (solar)
├── results/
│   └── plots.py               — 6 publication-quality PNG figures
├── tests/
│   └── test_all.py            — pytest unit tests (7 test classes)
├── main.py                    — Complete 9-step analysis runner
├── requirements.txt
└── README.md
```

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run full analysis
```bash
python main.py
```

### Run unit tests
```bash
python -m pytest tests/test_all.py -v
```

---

## Expected Results

After running `main.py`, the following are produced in `results/`:

| File | Description |
|------|-------------|
| `plot1_bus_inertia_distribution.png` | H_B_sync vs H_B_total per bus |
| `plot2_pareto_front.png` | NSGA-II Pareto: penetration vs losses vs SG count |
| `plot3_frequency_response.png` | Frequency trajectories: Bus 9 vs Bus 22 |
| `plot4_rocof_comparison.png` | Per-bus RoCoF without vs with GFM |
| `plot5_inertia_dynamics.png` | System inertia over time (Scenario C) |
| `plot6_min_vs_achieved.png` | H_min vs H_B_sync vs H_B_total |

**Key findings:**
- GFM inverters at buses 11 and 29 improve bus inertia across all 29 load buses
- Non-uniform inertia improvement (electrically close buses benefit most)
- NSGA-II identifies Pareto-optimal VRES dispatch schedules satisfying inertia constraints
- Bus 22 (low inertia) shows significantly worse frequency nadir than Bus 9 (high inertia)

---

## References

1. **Ghosh, S. et al.** (2023). "Assessment of Bus Inertia to Enhance Dynamic Flexibility of Hybrid Power Systems With Renewable Energy Integration." *IEEE Transactions on Power Delivery*, vol. 38, no. 4.

2. **Swetala, B. et al.** (2025). "Multi-Area Frequency Response Model with Virtual Inertia Integration for Optimal Variable Renewable Energy Scheduling." *IEEE Access*, vol. 13.

3. **Kundur, P.** (1994). *Power System Stability and Control*. McGraw-Hill.

4. **Bevrani, H. et al.** (2014). "Virtual Synchronous Generators: A Survey and New Perspectives." *Int. J. Electr. Power Energy Syst.*, vol. 54.

5. **Deb, K. et al.** (2002). "A Fast and Elitist Multi-Objective Genetic Algorithm: NSGA-II." *IEEE Trans. Evolutionary Computation*, vol. 6, no. 2.
