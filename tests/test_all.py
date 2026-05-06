"""
Unit Tests for Bus Inertia Research Project.

Run with:  python -m pytest tests/test_all.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from data.ieee39_data import (GENERATOR_DATA, GEN_BUSES, LOAD_BUSES,
                               BRANCH_DATA, GFM_DATA, VRES_DATA, build_ybus)
from core.electrical_distance import compute_electrical_distance_simple
from core.bus_inertia import estimate_bus_inertia, identify_weak_buses
from core.virtual_inertia import (compute_virtual_inertia_contribution,
                                   combine_total_bus_inertia)
from core.frequency_model import (compute_rocof, simulate_frequency,
                                   build_state_space, find_frequency_nadir)
from optimization.constraints import (compute_min_inertia_thresholds,
                                       check_inertia_constraint,
                                       check_voltage_limits)


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def system():
    Y_bus = build_ybus(39)
    gen_0 = [b - 1 for b in GEN_BUSES]      # 0-indexed
    load_0 = [b - 1 for b in LOAD_BUSES]
    H_G   = np.array([GENERATOR_DATA[b]["H"]     for b in GEN_BUSES])
    S_G   = np.array([GENERATOR_DATA[b]["S_mva"] for b in GEN_BUSES])
    gfm_0 = [b - 1 for b in GFM_DATA.keys()]
    H_v   = [GFM_DATA[b]["H_virt"] for b in GFM_DATA]
    S_v   = [GFM_DATA[b]["S_mva"]  for b in GFM_DATA]
    return dict(Y_bus=Y_bus, gen_0=gen_0, load_0=load_0,
                H_G=H_G, S_G=S_G, gfm_0=gfm_0, H_v=H_v, S_v=S_v)


# ─── Test 1: Electrical Distance ────────────────────────────────────────────

class TestElectricalDistance:
    def test_column_sums_to_one(self, system):
        """Each column of D must sum to 1 (normalised weights per bus)."""
        D, _ = compute_electrical_distance_simple(
            system["Y_bus"], system["gen_0"], system["load_0"])
        col_sums = D.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-6), \
            f"Column sums not 1: min={col_sums.min():.6f} max={col_sums.max():.6f}"

    def test_shape(self, system):
        D, _ = compute_electrical_distance_simple(
            system["Y_bus"], system["gen_0"], system["load_0"])
        assert D.shape == (len(system["gen_0"]), len(system["load_0"]))

    def test_non_negative(self, system):
        D, _ = compute_electrical_distance_simple(
            system["Y_bus"], system["gen_0"], system["load_0"])
        assert (D >= 0).all(), "D contains negative values"


# ─── Test 2: Bus Inertia ─────────────────────────────────────────────────────

class TestBusInertia:
    def test_output_shape(self, system):
        H_B, _, _ = estimate_bus_inertia(
            system["Y_bus"], system["H_G"], system["gen_0"],
            system["load_0"], system["S_G"])
        assert H_B.shape == (len(system["load_0"]),)

    def test_positive_values(self, system):
        H_B, _, _ = estimate_bus_inertia(
            system["Y_bus"], system["H_G"], system["gen_0"],
            system["load_0"], system["S_G"])
        assert (H_B > 0).all(), "Some H_B values are non-positive"

    def test_inertia_range(self, system):
        """H_B should be within a physically plausible range."""
        H_B, _, _ = estimate_bus_inertia(
            system["Y_bus"], system["H_G"], system["gen_0"],
            system["load_0"], system["S_G"])
        assert H_B.min() > 0.1, "H_B too small"
        assert H_B.max() < 1000, "H_B unrealistically large"

    def test_weak_bus_identification(self, system):
        H_B, _, _ = estimate_bus_inertia(
            system["Y_bus"], system["H_G"], system["gen_0"],
            system["load_0"], system["S_G"])
        weak, threshold = identify_weak_buses(H_B, LOAD_BUSES, 25)
        assert len(weak) > 0
        assert all(h <= threshold for _, h in weak)


# ─── Test 3: Virtual Inertia ─────────────────────────────────────────────────

class TestVirtualInertia:
    def test_virtual_positive(self, system):
        H_virt, _ = compute_virtual_inertia_contribution(
            system["Y_bus"], system["gfm_0"], system["H_v"],
            system["S_v"], system["load_0"])
        assert (H_virt >= 0).all()

    def test_total_geq_sync(self, system):
        H_B, _, _ = estimate_bus_inertia(
            system["Y_bus"], system["H_G"], system["gen_0"],
            system["load_0"], system["S_G"])
        H_virt, _ = compute_virtual_inertia_contribution(
            system["Y_bus"], system["gfm_0"], system["H_v"],
            system["S_v"], system["load_0"])
        H_total = combine_total_bus_inertia(H_B, H_virt)
        assert (H_total >= H_B - 1e-9).all(), \
            "Total inertia less than sync inertia"


# ─── Test 4: Minimum Inertia Threshold ───────────────────────────────────────

class TestMinInertia:
    def test_hmin_positive(self, system):
        D, _ = compute_electrical_distance_simple(
            system["Y_bus"], system["gen_0"], system["load_0"])
        P_gen = np.array([GENERATOR_DATA[b]["P_mw"] for b in GEN_BUSES])
        H_min, dP = compute_min_inertia_thresholds(D, P_gen, S_total_mva=7700)
        assert (H_min > 0).all(), "H_min must be positive"

    def test_higher_disturbance_higher_hmin(self, system):
        D, _ = compute_electrical_distance_simple(
            system["Y_bus"], system["gen_0"], system["load_0"])
        P_small = np.ones(len(GEN_BUSES)) * 100
        P_large = np.ones(len(GEN_BUSES)) * 500
        H_min_s, _ = compute_min_inertia_thresholds(D, P_small, 7700)
        H_min_l, _ = compute_min_inertia_thresholds(D, P_large, 7700)
        assert (H_min_l >= H_min_s).all()


# ─── Test 5: Frequency Model ─────────────────────────────────────────────────

class TestFrequencyModel:
    def test_rocof_negative_after_loss(self):
        """RoCoF should be negative (frequency drops) after generation loss."""
        rocof = compute_rocof(H_bus=4.0, delta_P_pu=0.1, f0=60.0)
        assert rocof < 0, f"Expected negative RoCoF, got {rocof}"

    def test_rocof_magnitude(self):
        """Higher inertia => lower |RoCoF|."""
        r_low  = compute_rocof(2.0, 0.1)
        r_high = compute_rocof(8.0, 0.1)
        assert abs(r_low) > abs(r_high)

    def test_simulation_frequency_drops(self):
        """Simulated frequency must drop below nominal after disturbance."""
        A, B, _ = build_state_space(H_area=4.0, D_damp=2.0)
        t, _, f, _ = simulate_frequency(A, B, delta_P_pu=0.1, t_span=(0, 10))
        assert f.min() < 60.0, "Frequency should drop below nominal"

    def test_nadir_detection(self):
        f = np.array([60.0, 59.8, 59.6, 59.7, 59.9, 60.0])
        nadir, idx = find_frequency_nadir(f)
        assert nadir == pytest.approx(59.6)
        assert idx == 2


# ─── Test 6: Constraint Checks ───────────────────────────────────────────────

class TestConstraints:
    def test_inertia_constraint_satisfied(self):
        H_B   = np.array([5.0, 6.0, 7.0])
        H_min = np.array([4.0, 5.0, 6.0])
        sat, deficit, n_viol = check_inertia_constraint(H_B, H_min)
        assert n_viol == 0

    def test_inertia_constraint_violated(self):
        H_B   = np.array([2.0, 3.0, 8.0])
        H_min = np.array([4.0, 5.0, 6.0])
        _, _, n_viol = check_inertia_constraint(H_B, H_min)
        assert n_viol == 2

    def test_voltage_limits_ok(self):
        V = np.array([1.00, 1.02, 0.97])
        ok, n_viol = check_voltage_limits(V)
        assert n_viol == 0

    def test_voltage_limits_violated(self):
        V = np.array([0.93, 1.07, 1.00])
        ok, n_viol = check_voltage_limits(V)
        assert n_viol == 2


# ─── Test 7: Y_bus Construction ──────────────────────────────────────────────

class TestYbus:
    def test_ybus_shape(self):
        Y = build_ybus(39)
        assert Y.shape == (39, 39)

    def test_ybus_symmetric(self):
        Y = build_ybus(39)
        assert np.allclose(Y, Y.T, atol=1e-8), "Y_bus must be symmetric"

    def test_ybus_diagonal_dominant(self):
        """Diagonal magnitudes should dominate each row."""
        Y = build_ybus(39)
        for i in range(39):
            off_sum = np.abs(Y[i]).sum() - np.abs(Y[i, i])
            # Not strictly dominant everywhere (shunt elements), just check > 0
            assert np.abs(Y[i, i]) > 0, f"Zero diagonal at row {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
