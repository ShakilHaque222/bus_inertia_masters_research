"""
Multi-Area Frequency Response Model.

Reference: Swetala et al., "Multi-Area Frequency Response with Virtual
           Inertia for Optimal VRES Scheduling," IEEE Access, 2025.
           Appendix A state-space formulation.

State vector per area i:
    X_i = [Delta_f_i, Delta_P_sg_i, Delta_P_G_i, Delta_P_S_i, Delta_delta_i]

where:
    Delta_f_i    = frequency deviation [pu Hz]
    Delta_P_sg_i = synchronous generator output deviation [pu MW]
    Delta_P_G_i  = governor output [pu MW]
    Delta_P_S_i  = storage/VSG power output [pu MW]
    Delta_delta_i= tie-line angle [rad]
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def build_state_space(H_area, D_damp, tau_t=0.3, tau_g=0.1,
                      R=0.05, K_s=0.1, T_s=10.0, T_tie=0.0873):
    """
    Build single-area state-space matrices A, B, C.

    Parameters
    ----------
    H_area  : float, area inertia constant [s]
    D_damp  : float, load damping coefficient [pu MW / pu Hz]
    tau_t   : float, turbine time constant [s]
    tau_g   : float, governor time constant [s]
    R       : float, speed droop [pu Hz / pu MW]
    K_s     : float, storage control gain [pu]
    T_s     : float, storage time constant [s]
    T_tie   : float, tie-line synchronising coefficient [pu MW/rad]

    Returns
    -------
    A : np.ndarray (5x5)
    B : np.ndarray (5x1)  — input = delta_P_load disturbance
    C : np.ndarray (1x5)  — output = delta_f
    """
    # Swing equation: 2H * d(df)/dt = delta_P_sg + delta_P_S - D*df - delta_P_L
    # Governor:       tau_g * d(dP_G)/dt = -df/R - dP_G
    # Turbine:        tau_t * d(dP_sg)/dt = dP_G - dP_sg
    # Storage/VSG:    T_s   * d(dP_S)/dt  = K_s*(-df) - dP_S
    # Tie-line:       d(d_delta)/dt = 2*pi*f0 * df

    f0 = 60.0
    A = np.array([
        # df              dP_sg          dP_G           dP_S          d_delta
        [-D_damp/(2*H_area), 1/(2*H_area), 0,             1/(2*H_area), -T_tie/(2*H_area)],
        [0,                  -1/tau_t,     1/tau_t,       0,             0               ],
        [-1/(R*tau_g),       0,            -1/tau_g,      0,             0               ],
        [-K_s/T_s,           0,            0,             -1/T_s,        0               ],
        [2*np.pi*f0,         0,            0,             0,             0               ],
    ])

    B = np.array([[-1/(2*H_area)], [0], [0], [0], [0]])  # disturbance input

    C = np.array([[1, 0, 0, 0, 0]])   # output = delta_f

    return A, B, C


def compute_rocof(H_bus, delta_P_pu, f0=60.0):
    """
    Compute initial RoCoF from swing equation.

    RoCoF = -delta_P_pu * f0 / (2 * H_bus)   [Hz/s]

    Parameters
    ----------
    H_bus      : float or array, bus inertia [s]
    delta_P_pu : float, power imbalance [pu] (positive = generation loss)
    f0         : float, nominal frequency [Hz]

    Returns
    -------
    rocof : float or array [Hz/s]  (negative = frequency drop)
    """
    H_bus = np.asarray(H_bus, dtype=float)
    H_bus = np.where(H_bus < 1e-6, 1e-6, H_bus)
    return -delta_P_pu * f0 / (2.0 * H_bus)


def simulate_frequency(A, B, delta_P_pu, t_span=(0, 20), dt=0.01):
    """
    Simulate frequency response via state-space ODE.

    dx/dt = A*x + B*u(t),   u(t) = delta_P_pu  for t >= t_fault
    Initial state: x0 = zeros

    Parameters
    ----------
    A          : np.ndarray (5x5)
    B          : np.ndarray (5x1)
    delta_P_pu : float, power imbalance [pu]
    t_span     : tuple (t_start, t_end)
    dt         : float

    Returns
    -------
    t      : np.ndarray
    x      : np.ndarray (5 x len(t))
    f_hz   : np.ndarray, frequency trajectory [Hz]
    rocof  : np.ndarray [Hz/s]
    """
    t_fault = 1.0
    f0 = 60.0

    def ode(t, x):
        u = delta_P_pu if t >= t_fault else 0.0
        return A @ x + B.flatten() * u

    t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / dt) + 1)
    sol = solve_ivp(ode, t_span, np.zeros(A.shape[0]),
                    t_eval=t_eval, method="RK45",
                    rtol=1e-6, atol=1e-8)

    t   = sol.t
    x   = sol.y
    f_hz    = f0 + x[0, :] * f0           # df in pu -> absolute Hz
    rocof   = np.gradient(f_hz, t)
    return t, x, f_hz, rocof


def find_frequency_nadir(f_trajectory):
    """
    Find the minimum frequency (nadir) in a trajectory.

    Parameters
    ----------
    f_trajectory : np.ndarray, frequency in Hz

    Returns
    -------
    nadir     : float, minimum frequency [Hz]
    nadir_idx : int, index of nadir
    """
    nadir_idx = int(np.argmin(f_trajectory))
    return float(f_trajectory[nadir_idx]), nadir_idx


def compute_area_rocof(H_B_array, delta_P_mw, S_total_mva, f0=60.0):
    """
    Compute per-bus RoCoF after a disturbance of delta_P_mw MW.

    Parameters
    ----------
    H_B_array   : np.ndarray, bus inertia [s]
    delta_P_mw  : float, power loss [MW]
    S_total_mva : float, total system capacity [MVA]
    f0          : float

    Returns
    -------
    rocof_array : np.ndarray [Hz/s]
    """
    delta_P_pu = delta_P_mw / S_total_mva
    return compute_rocof(H_B_array, delta_P_pu, f0)
