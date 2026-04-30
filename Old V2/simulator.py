"""
simulator.py
============
Python port of the 3-segment planar jump model.

Changes from previous version:
  - Torque profile changed from sigmoidal bell curve to STEP FUNCTION:
      torque is applied at full strength from t=0 until liftoff.
  - tau_max values are now FIXED at literature-based maxima (Harbo 2012,
    Hussain & Frey-Law 2016) and are NOT optimization parameters:
        ankle : 175 N·m  (plantarflexion, Hussain & Frey-Law 2016)
        knee  : 247 N·m  (extension, Harbo et al. 2012)
        hip   : 175 N·m  (extension, Harbo et al. 2012)
  - simulate_jump() now accepts a 3-element params_vec [theta1, theta2, theta3]
    (initial joint angles only). Torques start at t=0 and run until liftoff.

Model (unchanged):
  - 3-segment inverted pendulum: foot, shank, thigh
  - Toe pinned at origin during push-off (Phase 1)
  - Free ballistic flight after liftoff (Phase 2)
  - Equations of motion via Jacobian method

Parameters (Winter 2009, 70 kg / 1.75 m male):
  Segment masses, lengths, COM locations, moments of inertia
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve


# ─────────────────────────────────────────────────────────────────────────────
#  FIXED TORQUE MAXIMA  (literature-based, Harbo 2012; Hussain & Frey-Law 2016)
# ─────────────────────────────────────────────────────────────────────────────

TAU_MAX_ANKLE = 175.0   # N·m  plantarflexion (Hussain & Frey-Law, J Foot Ankle Res 2016)
TAU_MAX_KNEE  = 247.0   # N·m  knee extension (Harbo et al., Eur J Appl Physiol 2012)
TAU_MAX_HIP   = 175.0   # N·m  hip extension  (Harbo et al., Eur J Appl Physiol 2012)

TAU_MAX_FIXED = np.array([TAU_MAX_ANKLE, TAU_MAX_KNEE, TAU_MAX_HIP])


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PARAMETERS  (Winter 2009, 70 kg / 1.75 m male)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_params():
    """
    Returns anthropometric model parameters from Winter (2009).
    All values scaled to a 70 kg, 1.75 m male subject.
    """
    body_mass   = 70.0
    body_height = 1.75

    p = {}
    p['g']     = 9.81
    p['m1']    = 0.0145 * body_mass    # foot   (1.015 kg)
    p['m2']    = 0.0465 * body_mass    # shank  (3.255 kg)
    p['m3']    = 0.100  * body_mass    # thigh  (7.000 kg)
    p['I_HAT'] = 0.678  * body_mass * (0.1)**2   # HAT rotational inertia at hip

    p['L1'] = 0.055 * body_height      # foot length  (0.096 m)
    p['L2'] = 0.246 * body_height      # shank length (0.431 m)
    p['L3'] = 0.245 * body_height      # thigh length (0.429 m)

    p['d1'] = 0.500 * p['L1']          # foot  COM from proximal
    p['d2'] = 0.433 * p['L2']          # shank COM from proximal
    p['d3'] = 0.433 * p['L3']          # thigh COM from proximal

    p['I1'] = p['m1'] * (0.475 * p['L1'])**2
    p['I2'] = p['m2'] * (0.302 * p['L2'])**2
    p['I3'] = p['m3'] * (0.323 * p['L3'])**2

    return p


# ─────────────────────────────────────────────────────────────────────────────
#  TORQUE PROFILE  —  STEP FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_torques(t, tau_max=None):
    """
    Step-function torque profile.

    Torques are applied at full magnitude from t=0 and held constant until
    the liftoff event terminates Phase 1.  This replaces the previous
    sigmoidal ramp-up / ramp-down profile.

    tau_sign = [-1, -1, +1]  (ankle, knee, hip) — unchanged from before.

    Parameters
    ----------
    t : float
        Current simulation time (not used by step function, kept for
        compatibility with solve_ivp interface).
    tau_max : array-like, shape (3,), optional
        Peak torques [ankle, knee, hip] in N·m.
        Defaults to TAU_MAX_FIXED (literature values).

    Returns
    -------
    tau : np.ndarray, shape (3,)
    """
    if tau_max is None:
        tau_max = TAU_MAX_FIXED
    tau_sign = np.array([-1.0, -1.0, 1.0])
    return tau_sign * np.asarray(tau_max)


# ─────────────────────────────────────────────────────────────────────────────
#  JACOBIAN MASS MATRIX  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def mass_matrix(q, p):
    """3×3 generalised inertia matrix via Jacobian method."""
    a1 = q[0]
    a2 = q[0] + q[1]
    a3 = q[0] + q[1] + q[2]

    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    I1, I2, I3 = p['I1'], p['I2'], p['I3']
    I_HAT      = p['I_HAT']

    J1 = np.array([[ d1*np.cos(a1),               0,            0],
                   [-d1*np.sin(a1),               0,            0]])
    J2 = np.array([[ L1*np.cos(a1)+d2*np.cos(a2),  d2*np.cos(a2),  0],
                   [-L1*np.sin(a1)-d2*np.sin(a2), -d2*np.sin(a2),  0]])
    J3 = np.array([[ L1*np.cos(a1)+L2*np.cos(a2)+d3*np.cos(a3),
                     L2*np.cos(a2)+d3*np.cos(a3),
                     d3*np.cos(a3)],
                   [-L1*np.sin(a1)-L2*np.sin(a2)-d3*np.sin(a3),
                    -L2*np.sin(a2)-d3*np.sin(a3),
                    -d3*np.sin(a3)]])

    Jw1 = np.array([[1, 0, 0]], dtype=float)
    Jw2 = np.array([[1, 1, 0]], dtype=float)
    Jw3 = np.array([[1, 1, 1]], dtype=float)

    M = (m1 * J1.T @ J1 + I1 * Jw1.T @ Jw1 +
         m2 * J2.T @ J2 + I2 * Jw2.T @ Jw2 +
         m3 * J3.T @ J3 + (I3 + I_HAT) * Jw3.T @ Jw3)
    return M


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 ODE  (toe-pinned push-off)  — unchanged except torque call
# ─────────────────────────────────────────────────────────────────────────────

def jump_ode_phase1(t, X, p, tau_max=None):
    """Toe-pinned 3-segment dynamics. EOM: M*ddq = tau - h - G."""
    q  = X[:3]
    dq = X[3:]

    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    g          = p['g']

    a1 = q[0];  a2 = q[0]+q[1];  a3 = q[0]+q[1]+q[2]
    da1 = dq[0]; da2 = dq[0]+dq[1]; da3 = dq[0]+dq[1]+dq[2]

    M = mass_matrix(q, p)

    J1 = np.array([[ d1*np.cos(a1),               0,  0],
                   [-d1*np.sin(a1),               0,  0]])
    J2 = np.array([[ L1*np.cos(a1)+d2*np.cos(a2),  d2*np.cos(a2),  0],
                   [-L1*np.sin(a1)-d2*np.sin(a2), -d2*np.sin(a2),  0]])
    J3 = np.array([[ L1*np.cos(a1)+L2*np.cos(a2)+d3*np.cos(a3),
                     L2*np.cos(a2)+d3*np.cos(a3),  d3*np.cos(a3)],
                   [-L1*np.sin(a1)-L2*np.sin(a2)-d3*np.sin(a3),
                    -L2*np.sin(a2)-d3*np.sin(a3), -d3*np.sin(a3)]])

    dJ1dt = np.array([[-d1*np.sin(a1)*da1,  0,  0],
                      [-d1*np.cos(a1)*da1,  0,  0]])
    dJ2dt = np.array([[-L1*np.sin(a1)*da1-d2*np.sin(a2)*da2, -d2*np.sin(a2)*da2, 0],
                      [-L1*np.cos(a1)*da1-d2*np.cos(a2)*da2, -d2*np.cos(a2)*da2, 0]])
    dJ3dt = np.array([
        [-L1*np.sin(a1)*da1-L2*np.sin(a2)*da2-d3*np.sin(a3)*da3,
         -L2*np.sin(a2)*da2-d3*np.sin(a3)*da3, -d3*np.sin(a3)*da3],
        [-L1*np.cos(a1)*da1-L2*np.cos(a2)*da2-d3*np.cos(a3)*da3,
         -L2*np.cos(a2)*da2-d3*np.cos(a3)*da3, -d3*np.cos(a3)*da3]
    ])

    h = (m1 * J1.T @ (dJ1dt @ dq) +
         m2 * J2.T @ (dJ2dt @ dq) +
         m3 * J3.T @ (dJ3dt @ dq))

    G = np.zeros(3)
    G[0] += m1 * g * (-d1*np.sin(a1))
    G[0] += m2 * g * (-L1*np.sin(a1) - d2*np.sin(a2))
    G[1] += m2 * g * (-d2*np.sin(a2))
    G[0] += m3 * g * (-L1*np.sin(a1) - L2*np.sin(a2) - d3*np.sin(a3))
    G[1] += m3 * g * (-L2*np.sin(a2) - d3*np.sin(a3))
    G[2] += m3 * g * (-d3*np.sin(a3))

    tau = compute_torques(t, tau_max)
    ddq = solve(M, tau - h - G)

    return np.concatenate([dq, ddq])


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 ODE  (free-body flight)  — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def jump_ode_phase2(t, X, p):
    """Free-body ballistic flight."""
    vx_com = X[5]; vy_com = X[6]
    dtheta = X[7:10]
    return np.array([vx_com, vy_com,
                     dtheta[0], dtheta[1], dtheta[2],
                     0.0, -p['g'],
                     0.0, 0.0, 0.0])


# ─────────────────────────────────────────────────────────────────────────────
#  COM HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_com_position(X, p):
    a1 = X[0]; a2 = X[0]+X[1]; a3 = X[0]+X[1]+X[2]
    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    M_tot = m1 + m2 + m3

    x1 = d1*np.sin(a1);  y1 = d1*np.cos(a1)
    x2 = L1*np.sin(a1) + d2*np.sin(a2);  y2 = L1*np.cos(a1) + d2*np.cos(a2)
    x3 = L1*np.sin(a1) + L2*np.sin(a2) + d3*np.sin(a3)
    y3 = L1*np.cos(a1) + L2*np.cos(a2) + d3*np.cos(a3)

    return (m1*x1+m2*x2+m3*x3)/M_tot, (m1*y1+m2*y2+m3*y3)/M_tot


def compute_com_velocity(X, p):
    a1 = X[0]; a2 = X[0]+X[1]; a3 = X[0]+X[1]+X[2]
    dp1 = X[3]; dp2 = X[3]+X[4]; dp3 = X[3]+X[4]+X[5]
    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    M_tot = m1 + m2 + m3

    vx1 =  d1*dp1*np.cos(a1);  vy1 = -d1*dp1*np.sin(a1)
    vx2 =  L1*dp1*np.cos(a1) + d2*dp2*np.cos(a2)
    vy2 = -L1*dp1*np.sin(a1) - d2*dp2*np.sin(a2)
    vx3 =  L1*dp1*np.cos(a1) + L2*dp2*np.cos(a2) + d3*dp3*np.cos(a3)
    vy3 = -L1*dp1*np.sin(a1) - L2*dp2*np.sin(a2) - d3*dp3*np.sin(a3)

    return (m1*vx1+m2*vx2+m3*vx3)/M_tot, (m1*vy1+m2*vy2+m3*vy3)/M_tot


def compute_com_accel_y(X, ddtheta, p):
    a1 = X[0]; a2 = X[0]+X[1]; a3 = X[0]+X[1]+X[2]
    dp1 = X[3]; dp2 = X[3]+X[4]; dp3 = X[3]+X[4]+X[5]
    ddp1 = ddtheta[0]; ddp2 = ddtheta[0]+ddtheta[1]; ddp3 = ddtheta[0]+ddtheta[1]+ddtheta[2]
    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    M_tot = m1 + m2 + m3

    ay1 = -ddp1*d1*np.sin(a1) - dp1**2*d1*np.cos(a1)
    ay2 = (-ddp1*L1*np.sin(a1) - dp1**2*L1*np.cos(a1)
           -ddp2*d2*np.sin(a2) - dp2**2*d2*np.cos(a2))
    ay3 = (-ddp1*L1*np.sin(a1) - dp1**2*L1*np.cos(a1)
           -ddp2*L2*np.sin(a2) - dp2**2*L2*np.cos(a2)
           -ddp3*d3*np.sin(a3) - dp3**2*d3*np.cos(a3))

    return (m1*ay1 + m2*ay2 + m3*ay3) / M_tot


# ─────────────────────────────────────────────────────────────────────────────
#  LIFTOFF EVENT  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def make_liftoff_event(p, tau_max=None):
    """Event function: triggers when GRF_y → 0 (toe leaves ground)."""
    def liftoff_event(t, X):
        dX      = jump_ode_phase1(t, X, p, tau_max)
        ddtheta = dX[3:]
        ay      = compute_com_accel_y(X, ddtheta, p)
        M_tot   = p['m1'] + p['m2'] + p['m3']
        GRF_y   = M_tot * (p['g'] + ay)
        return GRF_y

    liftoff_event.terminal  = True
    liftoff_event.direction = -1
    return liftoff_event


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SIMULATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_jump(params_vec, p=None, return_trajectories=False, verbose=False):
    """
    Run the full jump simulation and return jump height.

    Parameters
    ----------
    params_vec : array-like, shape (3,)
        [theta1, theta2, theta3]  — initial joint angles (rad).
        Torques are fixed at literature maxima and applied as a step from t=0.

    p : dict, optional
        Model parameters. Uses get_default_params() if None.

    return_trajectories : bool
        If True, also returns (t1, X1, t2, X2).

    verbose : bool
        Print simulation results to console.

    Returns
    -------
    jump_height : float  (meters)
    """
    if p is None:
        p = get_default_params()

    params_vec = np.asarray(params_vec, dtype=float)
    theta0 = params_vec[0:3]

    # Fixed literature-based torques (step function, applied from t=0)
    tau_max = TAU_MAX_FIXED.copy()

    # ── Initial conditions ────────────────────────────────────────────────────
    X0_p1 = np.concatenate([theta0, np.zeros(3)])

    _, y_com_init = compute_com_position(X0_p1, p)
    if y_com_init <= 0:
        if verbose:
            print(f'Infeasible: COM below ground (y={y_com_init:.4f} m)')
        return (0.0, None, None, None, None) if return_trajectories else 0.0

    # ── Phase 1: toe-pinned push-off ──────────────────────────────────────────
    liftoff_event = make_liftoff_event(p, tau_max)

    try:
        sol1 = solve_ivp(
            fun      = lambda t, X: jump_ode_phase1(t, X, p, tau_max),
            t_span   = (0.0, 1.0),
            y0       = X0_p1,
            method   = 'RK45', # Can be Radau (Better, but more computationally expensive), or RK45 (Faster, but has less precision at more than 4 degrees)
            rtol     = 1e-6,
            atol     = 1e-8,
            # max_step = 1e-4, # Unnecessary for RK45
            events   = liftoff_event,
        )
    except Exception as e:
        if verbose:
            print(f'Phase 1 solver failed: {e}')
        return (0.0, None, None, None, None) if return_trajectories else 0.0

    if not sol1.t_events[0].size or sol1.t_events[0][0] < 0.005:
        if verbose:
            print('Liftoff not detected or too early.')
        return (0.0, None, None, None, None) if return_trajectories else 0.0

    t1  = sol1.t
    X1  = sol1.y.T
    te1 = sol1.t_events[0][0]

    X_lo = X1[-1]
    x_com_lo, y_com_lo   = compute_com_position(X_lo, p)
    vx_com_lo, vy_com_lo = compute_com_velocity(X_lo, p)

    if vy_com_lo <= 0:
        if verbose:
            print(f'Liftoff velocity negative: vy={vy_com_lo:.4f} m/s')
        return (0.0, t1, X1, None, None) if return_trajectories else 0.0

    if verbose:
        print(f'Liftoff at t={te1:.4f} s')
        print(f'  COM position: ({x_com_lo:.4f}, {y_com_lo:.4f}) m')
        print(f'  COM velocity: ({vx_com_lo:.4f}, {vy_com_lo:.4f}) m/s')

    # ── Phase 2: ballistic flight ─────────────────────────────────────────────
    X0_p2 = np.array([
        x_com_lo, y_com_lo,
        X_lo[0], X_lo[1], X_lo[2],
        vx_com_lo, vy_com_lo,
        0.0, 0.0, 0.0
    ])

    try:
        sol2 = solve_ivp(
            fun      = lambda t, X: jump_ode_phase2(t, X, p),
            t_span   = (te1, te1 + 2.0),
            y0       = X0_p2,
            method   = 'RK45',
            rtol     = 1e-8,
            atol     = 1e-10,
            max_step = 1e-3,
        )
    except Exception as e:
        if verbose:
            print(f'Phase 2 solver failed: {e}')
        return (0.0, t1, X1, None, None) if return_trajectories else 0.0

    t2 = sol2.t
    X2 = sol2.y.T

    jump_height = float(np.max(X2[:, 1]) - y_com_lo)

    if verbose:
        print(f'Jump height: {jump_height:.4f} m ({jump_height*100:.2f} cm)')

    if return_trajectories:
        return jump_height, t1, X1, t2, X2
    return jump_height


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 3-parameter interface: only starting joint angles
    params_vec = np.array([-0.1, 0.6, -1.1])

    print('Running simulation with step-function torques...')
    print(f'  tau_max = [ankle={TAU_MAX_ANKLE}, knee={TAU_MAX_KNEE}, hip={TAU_MAX_HIP}] N·m')
    h, t1, X1, t2, X2 = simulate_jump(
        params_vec,
        return_trajectories=True,
        verbose=True
    )
    print(f'\nResult: jump_height = {h:.4f} m ({h*100:.2f} cm)')
