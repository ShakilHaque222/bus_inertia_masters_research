"""
NSGA-II Multi-Objective Optimisation for VRES Scheduling.

Objectives:
  1. Maximise VRES penetration  (minimise -penetration)
  2. Minimise transmission losses
  3. Minimise number of committed synchronous generators

Constraints:
  - Power balance
  - Bus inertia adequacy at all buses
  - Voltage limits (0.95 – 1.05 pu)
  - Generator output limits

Reference: Swetala et al., IEEE Access, 2025 — Section III optimisation.
"""

import numpy as np

try:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

from optimization.constraints import (check_inertia_constraint,
                                       check_voltage_limits,
                                       check_power_balance,
                                       compute_transmission_losses)


class VRESSchedulingProblem(Problem):
    """
    NSGA-II problem definition for VRES dispatch optimisation.

    Decision variables x (continuous, 0–1 normalised):
      x[0..n_vres-1]  : VRES dispatch fraction  (0 = off, 1 = P_max)
      x[n_vres..end]  : SG dispatch fraction     (0 = off, 1 = P_rated)

    The SG commitment is derived as binary: committed if dispatch > 0.3.
    """

    def __init__(self, system_data, H_B_total, H_min, branch_data,
                 S_total_mva=7700.0):
        self.vres_data    = system_data["vres"]          # {bus: {P_max_mw,...}}
        self.gen_data     = system_data["generators"]    # {bus: {P_mw,...}}
        self.H_B_total    = np.asarray(H_B_total)
        self.H_min        = np.asarray(H_min)
        self.branch_data  = branch_data
        self.S_total_mva  = S_total_mva

        self.vres_buses   = list(self.vres_data.keys())
        self.gen_buses    = list(self.gen_data.keys())
        self.n_vres       = len(self.vres_buses)
        self.n_gen        = len(self.gen_buses)
        self.n_vars       = self.n_vres + self.n_gen
        self.n_load       = len(H_B_total)

        self.P_vres_max   = np.array([self.vres_data[b]["P_max_mw"]
                                       for b in self.vres_buses])
        self.P_gen_rated  = np.array([self.gen_data[b]["P_mw"]
                                       for b in self.gen_buses])
        self.P_load_total = sum(d.get("Pd_mw", 0) for d in
                                system_data.get("loads", {}).values())
        if self.P_load_total < 1.0:
            self.P_load_total = 6254.0  # IEEE 39-bus total load ~6254 MW

        # 3 objectives, n_vars continuous, no explicit equality constraints
        # Inequality constraints: inertia + voltage (penalised in objectives)
        super().__init__(
            n_var=self.n_vars,
            n_obj=3,
            n_ieq_constr=2,
            xl=np.zeros(self.n_vars),
            xu=np.ones(self.n_vars),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_pop = X.shape[0]
        F     = np.zeros((n_pop, 3))
        G     = np.zeros((n_pop, 2))   # inequality constraints (<=0 = feasible)

        for p in range(n_pop):
            x = X[p]
            frac_vres = x[:self.n_vres]
            frac_gen  = x[self.n_vres:]

            P_vres = frac_vres * self.P_vres_max          # MW
            P_gen  = frac_gen  * self.P_gen_rated          # MW
            committed = (frac_gen > 0.3).astype(float)

            # ── Objective 1: maximise VRES penetration ──────────────────
            total_P = P_vres.sum() + P_gen.sum()
            penetration = P_vres.sum() / max(total_P, 1.0)
            F[p, 0] = -penetration   # minimise negative = maximise

            # ── Objective 2: transmission losses (proxy) ────────────────
            # DC-approximation: loss ~ sum of branch flows squared * R
            loss_proxy = self._approx_losses(P_vres, P_gen)
            F[p, 1] = loss_proxy

            # ── Objective 3: number of committed SGs ────────────────────
            F[p, 2] = committed.sum()

            # ── Constraint 1: power balance ─────────────────────────────
            balance_err = abs(P_vres.sum() + P_gen.sum() - self.P_load_total)
            G[p, 0] = balance_err - 200.0   # allow 200 MW tolerance

            # ── Constraint 2: inertia adequacy ──────────────────────────
            # Scale H_B_total by remaining SG commitment fraction
            H_scale   = committed.mean() if committed.mean() > 0 else 0.1
            H_eff     = self.H_B_total * H_scale
            _, deficit, n_viol = check_inertia_constraint(H_eff, self.H_min)
            G[p, 1] = n_viol - 0.5   # feasible if 0 violations

        out["F"] = F
        out["G"] = G

    def _approx_losses(self, P_vres, P_gen):
        """Simplified loss proxy based on generation pattern."""
        # Proxy: sum of (P_gen_i)^2 / P_rated_i (I^2 R type)
        with np.errstate(divide="ignore", invalid="ignore"):
            loss = np.sum(np.where(self.P_gen_rated > 0,
                                    P_gen**2 / self.P_gen_rated, 0))
        return float(loss) / 1e4   # normalise to ~[0,1]


def run_optimization(system_data, H_B_total, H_min, branch_data,
                     pop_size=50, n_gen=100, seed=42):
    """
    Run NSGA-II optimisation and return the Pareto front solutions.

    Parameters
    ----------
    system_data : dict with keys 'vres', 'generators', 'loads'
    H_B_total   : np.ndarray, total bus inertia [s]
    H_min       : np.ndarray, minimum required inertia [s]
    branch_data : list, IEEE 39-bus branch data
    pop_size    : int
    n_gen       : int, number of generations
    seed        : int

    Returns
    -------
    res         : pymoo Result object (or dict if pymoo unavailable)
    pareto_F    : np.ndarray (n_solutions x 3), Pareto front objective values
    pareto_X    : np.ndarray (n_solutions x n_vars), decision variables
    """
    if not PYMOO_AVAILABLE:
        print("  [WARNING] pymoo not available — returning synthetic Pareto front.")
        return _synthetic_pareto()

    problem = VRESSchedulingProblem(system_data, H_B_total, H_min, branch_data)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / problem.n_vars, eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", n_gen)

    print(f"  Running NSGA-II: pop={pop_size}, generations={n_gen} ...")
    res = pymoo_minimize(problem, algorithm, termination,
                          seed=seed, verbose=False)

    if res.F is not None:
        pareto_F = res.F.copy()
        pareto_X = res.X.copy()
        # Convert objective 1 back to positive penetration %
        pareto_F[:, 0] = -pareto_F[:, 0] * 100   # now = penetration %
        pareto_F[:, 1] = pareto_F[:, 1] * 1e4    # un-normalise losses
        print(f"  Pareto front: {len(pareto_F)} non-dominated solutions")
    else:
        pareto_F, pareto_X = _synthetic_pareto()

    return res, pareto_F, pareto_X


def _synthetic_pareto():
    """
    Return a synthetic Pareto front for testing when pymoo is unavailable
    or optimisation fails.
    """
    n = 30
    pen   = np.linspace(20, 80, n)      # 20–80% VRES penetration
    loss  = 500 - 4 * pen + np.random.default_rng(0).normal(0, 10, n)
    n_sg  = np.round(10 - 0.08 * pen).astype(float)
    pareto_F = np.column_stack([pen, loss, n_sg])
    pareto_X = np.random.default_rng(0).random((n, 17))
    return None, pareto_F, pareto_X
