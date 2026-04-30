"""
simulator.py
============
Python port of the 3-segment planar jump model from MATLAB.

Mirrors jump_ode_phase1.m, jump_ode_phase2.m, and the simulate_jump()
function in run_simulation.m exactly, so results can be validated
side-by-side against the MATLAB implementation.

Model:
    - 3-segment inverted pendulum: foot, shank, thigh
    - Toe pinned at origin during push-off (Phase 1)
    - Free ballistic flight after liftoff (Phase 2)
    - Equations of motion via Jacobian method

Parameters (Winter 2009, 70 kg / 1.75 m male):
    Segment masses, lengths, COM locations, moments of inertia

Control (12 optimizable parameters for BOTorch):
    theta0  [3] - initial joint angles (rad)
    tau_max [3] - peak torques (Nm)
    t_on    [3] - activation onset times (s)
    t_dur   [3] - activation durations (s)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PARAMETERS  (Winter 2009, 70 kg / 1.75 m male)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_params():
    """
    Returns model parameters matching run_simulation.m exactly.
    All values from Winter (2009) anthropometric tables.
    """
    body_mass   = 70.0    # kg
    body_height = 1.75    # m

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

    p['I1'] = p['m1'] * (0.475 * p['L1'])**2   # foot  inertia about COM
    p['I2'] = p['m2'] * (0.302 * p['L2'])**2   # shank inertia about COM
    p['I3'] = p['m3'] * (0.323 * p['L3'])**2   # thigh inertia about COM

    return p


# ─────────────────────────────────────────────────────────────────────────────
#  TORQUE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def compute_torques(t, ctrl):
    """
    Sigmoidal ramp-up / ramp-down torque profile.

    tau_i(t) = tau_sign_i * tau_max_i * [sig(t, t_on_i) - sig(t, t_on_i + t_dur_i)]
    sig(t, t0) = 1 / (1 + exp(-k*(t - t0)))

    tau_sign = [-1, -1, +1]  (ankle, knee, hip)
    Matches compute_torques() in jump_ode_phase1.m
    """
    k        = ctrl['k']
    t_on     = ctrl['t_on']      # shape (3,)
    t_off    = t_on + ctrl['t_dur']
    tau_max  = ctrl['tau_max']   # shape (3,)
    tau_sign = np.array([-1.0, -1.0, 1.0])

    sig_on  = 1.0 / (1.0 + np.exp(-k * (t - t_on)))
    sig_off = 1.0 / (1.0 + np.exp(-k * (t - t_off)))

    return tau_sign * tau_max * (sig_on - sig_off)


# ─────────────────────────────────────────────────────────────────────────────
#  JACOBIAN MASS MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def mass_matrix(q, p):
    """
    3x3 generalised inertia matrix via Jacobian method.
    M = sum_i (m_i * Ji^T Ji + I_i * Jwi^T Jwi)

    Matches mass_matrix() in jump_ode_phase1.m
    """
    a1 = q[0]
    a2 = q[0] + q[1]
    a3 = q[0] + q[1] + q[2]

    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    I1, I2, I3 = p['I1'], p['I2'], p['I3']
    I_HAT      = p['I_HAT']

    # Translational Jacobians (2x3) for each segment COM
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

    # Angular Jacobians (1x3)
    Jw1 = np.array([[1, 0, 0]], dtype=float)
    Jw2 = np.array([[1, 1, 0]], dtype=float)
    Jw3 = np.array([[1, 1, 1]], dtype=float)

    M = (m1 * J1.T @ J1 + I1 * Jw1.T @ Jw1 +
         m2 * J2.T @ J2 + I2 * Jw2.T @ Jw2 +
         m3 * J3.T @ J3 + (I3 + I_HAT) * Jw3.T @ Jw3)

    return M


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 ODE  (toe-pinned push-off)
# ─────────────────────────────────────────────────────────────────────────────

def jump_ode_phase1(t, X, p, ctrl):
    """
    Toe-pinned 3-segment dynamics.
    State: X = [theta1, theta2, theta3, dtheta1, dtheta2, dtheta3]

    EOM: M(q)*ddq + h(q,dq) + G(q) = tau
    Solved as: ddq = M \ (tau - h - G)

    Matches jump_ode_phase1.m exactly.
    """
    q  = X[:3]
    dq = X[3:]

    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    g          = p['g']

    a1 = q[0];           a2 = q[0]+q[1];        a3 = q[0]+q[1]+q[2]
    da1 = dq[0];         da2 = dq[0]+dq[1];     da3 = dq[0]+dq[1]+dq[2]

    # ── Mass matrix ──────────────────────────────────────────────────────────
    M = mass_matrix(q, p)

    # ── Jacobians (needed for Coriolis) ──────────────────────────────────────
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

    # ── dJ/dt (time derivatives of Jacobians) ────────────────────────────────
    dJ1dt = np.array([[-d1*np.sin(a1)*da1,               0,  0],
                      [-d1*np.cos(a1)*da1,               0,  0]])

    dJ2dt = np.array([[-L1*np.sin(a1)*da1-d2*np.sin(a2)*da2, -d2*np.sin(a2)*da2, 0],
                      [-L1*np.cos(a1)*da1-d2*np.cos(a2)*da2, -d2*np.cos(a2)*da2, 0]])

    dJ3dt = np.array([
        [-L1*np.sin(a1)*da1-L2*np.sin(a2)*da2-d3*np.sin(a3)*da3,
         -L2*np.sin(a2)*da2-d3*np.sin(a3)*da3,
         -d3*np.sin(a3)*da3],
        [-L1*np.cos(a1)*da1-L2*np.cos(a2)*da2-d3*np.cos(a3)*da3,
         -L2*np.cos(a2)*da2-d3*np.cos(a3)*da3,
         -d3*np.cos(a3)*da3]
    ])

    # ── Coriolis vector: h = sum_i m_i * Ji^T * (dJi/dt * dq) ───────────────
    h = (m1 * J1.T @ (dJ1dt @ dq) +
         m2 * J2.T @ (dJ2dt @ dq) +
         m3 * J3.T @ (dJ3dt @ dq))

    # ── Gravity vector: G_k = dV/dq_k ────────────────────────────────────────
    G = np.zeros(3)
    # Segment 1 (foot)
    G[0] += m1 * g * (-d1*np.sin(a1))
    # Segment 2 (shank)
    G[0] += m2 * g * (-L1*np.sin(a1) - d2*np.sin(a2))
    G[1] += m2 * g * (-d2*np.sin(a2))
    # Segment 3 (thigh)
    G[0] += m3 * g * (-L1*np.sin(a1) - L2*np.sin(a2) - d3*np.sin(a3))
    G[1] += m3 * g * (-L2*np.sin(a2) - d3*np.sin(a3))
    G[2] += m3 * g * (-d3*np.sin(a3))

    # ── Torques ───────────────────────────────────────────────────────────────
    tau = compute_torques(t, ctrl)

    # ── EOM: M*ddq = tau - h - G ──────────────────────────────────────────────
    ddq = solve(M, tau - h - G)

    return np.concatenate([dq, ddq])


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 ODE  (free-body flight)
# ─────────────────────────────────────────────────────────────────────────────

def jump_ode_phase2(t, X, p):
    """
    Free-body ballistic flight.
    State: X = [x_com, y_com, theta1, theta2, theta3,
                vx_com, vy_com, dtheta1, dtheta2, dtheta3]

    COM follows ballistic trajectory. Segments hold angular velocity from liftoff.
    Matches jump_ode_phase2.m exactly.
    """
    vx_com = X[5]
    vy_com = X[6]
    dtheta = X[7:10]

    dx_com  = vx_com
    dy_com  = vy_com
    dvx_com = 0.0
    dvy_com = -p['g']
    ddtheta = np.zeros(3)

    return np.array([dx_com, dy_com,
                     dtheta[0], dtheta[1], dtheta[2],
                     dvx_com, dvy_com,
                     ddtheta[0], ddtheta[1], ddtheta[2]])


# ─────────────────────────────────────────────────────────────────────────────
#  COM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_com_position(X, p):
    """COM position from phase-1 state vector."""
    a1 = X[0]; a2 = X[0]+X[1]; a3 = X[0]+X[1]+X[2]
    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    M_tot = m1 + m2 + m3

    x1 = d1*np.sin(a1)
    y1 = d1*np.cos(a1)
    x2 = L1*np.sin(a1) + d2*np.sin(a2)
    y2 = L1*np.cos(a1) + d2*np.cos(a2)
    x3 = L1*np.sin(a1) + L2*np.sin(a2) + d3*np.sin(a3)
    y3 = L1*np.cos(a1) + L2*np.cos(a2) + d3*np.cos(a3)

    x_com = (m1*x1 + m2*x2 + m3*x3) / M_tot
    y_com = (m1*y1 + m2*y2 + m3*y3) / M_tot
    return x_com, y_com


def compute_com_velocity(X, p):
    """COM velocity from phase-1 state vector."""
    a1 = X[0]; a2 = X[0]+X[1]; a3 = X[0]+X[1]+X[2]
    dp1 = X[3]; dp2 = X[3]+X[4]; dp3 = X[3]+X[4]+X[5]
    m1, m2, m3 = p['m1'], p['m2'], p['m3']
    L1, L2     = p['L1'], p['L2']
    d1, d2, d3 = p['d1'], p['d2'], p['d3']
    M_tot = m1 + m2 + m3

    vx1 =  d1*dp1*np.cos(a1)
    vy1 = -d1*dp1*np.sin(a1)
    vx2 =  L1*dp1*np.cos(a1) + d2*dp2*np.cos(a2)
    vy2 = -L1*dp1*np.sin(a1) - d2*dp2*np.sin(a2)
    vx3 =  L1*dp1*np.cos(a1) + L2*dp2*np.cos(a2) + d3*dp3*np.cos(a3)
    vy3 = -L1*dp1*np.sin(a1) - L2*dp2*np.sin(a2) - d3*dp3*np.sin(a3)

    vx_com = (m1*vx1 + m2*vx2 + m3*vx3) / M_tot
    vy_com = (m1*vy1 + m2*vy2 + m3*vy3) / M_tot
    return vx_com, vy_com


def compute_com_accel_y(X, ddtheta, p):
    """Vertical COM acceleration — used in GRF liftoff event."""
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
#  LIFTOFF EVENT
# ─────────────────────────────────────────────────────────────────────────────

def make_liftoff_event(p, ctrl):
    """
    Returns a callable event function for solve_ivp.
    Triggers when GRF_y passes through zero (toe leaves ground).
    Matches liftoff_event() in run_simulation.m.
    """
    def liftoff_event(t, X):
        dX      = jump_ode_phase1(t, X, p, ctrl)
        ddtheta = dX[3:]
        ay      = compute_com_accel_y(X, ddtheta, p)
        M_tot   = p['m1'] + p['m2'] + p['m3']
        GRF_y   = M_tot * (p['g'] + ay)
        return GRF_y

    liftoff_event.terminal  = True
    liftoff_event.direction = -1   # GRF going from + to 0
    return liftoff_event


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SIMULATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_jump(params_vec, p=None, return_trajectories=False, verbose=False):
    """
    Run the full jump simulation and return jump height.

    Parameters
    ----------
    params_vec : array-like, shape (12,)
        [theta1, theta2, theta3,          <- initial angles (rad)
         tau_max_ankle, tau_max_knee, tau_max_hip,  <- peak torques (Nm)
         t_on_ankle, t_on_knee, t_on_hip,            <- onset times (s)
         t_dur_ankle, t_dur_knee, t_dur_hip]          <- durations (s)

    p : dict, optional
        Model parameters. Uses get_default_params() if None.

    return_trajectories : bool
        If True, also returns (t1, X1, t2, X2) for validation plotting.

    verbose : bool
        Print simulation results to console.

    Returns
    -------
    jump_height : float
        COM height gain above liftoff (m). Returns 0 for infeasible parameters.

    (optionally) t1, X1, t2, X2 : arrays
        Phase 1 and Phase 2 trajectories for validation.

    Matches simulate_jump() in run_simulation.m exactly.
    """
    if p is None:
        p = get_default_params()

    params_vec = np.asarray(params_vec, dtype=float)

    # ── Unpack 12 parameters ─────────────────────────────────────────────────
    theta0  = params_vec[0:3]
    tau_max = params_vec[3:6]
    t_on    = params_vec[6:9]
    t_dur   = params_vec[9:12]

    ctrl = {
        'tau_max': tau_max,
        't_on':    t_on,
        't_dur':   t_dur,
        'k':       50.0
    }

    # ── Initial conditions ────────────────────────────────────────────────────
    X0_p1 = np.concatenate([theta0, np.zeros(3)])

    # ── Feasibility: COM must be above ground ─────────────────────────────────
    _, y_com_init = compute_com_position(X0_p1, p)
    if y_com_init <= 0:
        if verbose:
            print(f'Infeasible: initial COM below ground (y={y_com_init:.4f} m)')
        return (0.0, None, None, None, None) if return_trajectories else 0.0

    # ── Phase 1: toe-pinned push-off ──────────────────────────────────────────
    liftoff_event = make_liftoff_event(p, ctrl)

    try:
        sol1 = solve_ivp(
            fun       = lambda t, X: jump_ode_phase1(t, X, p, ctrl),
            t_span    = (0.0, 1.0),
            y0        = X0_p1,
            method    = 'Radau',          # stiff solver, equivalent to ode15s
            rtol      = 1e-6,
            atol      = 1e-8,
            max_step  = 1e-4,
            events    = liftoff_event,
            dense_output = False
        )
    except Exception as e:
        if verbose:
            print(f'Phase 1 solver failed: {e}')
        return (0.0, None, None, None, None) if return_trajectories else 0.0

    # Check liftoff detected
    if not sol1.t_events[0].size or sol1.t_events[0][0] < 0.005:
        if verbose:
            print('Liftoff not detected or too early.')
        return (0.0, None, None, None, None) if return_trajectories else 0.0

    t1   = sol1.t
    X1   = sol1.y.T          # shape (n_steps, 6)
    te1  = sol1.t_events[0][0]

    # ── Liftoff state ──────────────────────────────────────────────────────────
    X_lo = X1[-1]
    x_com_lo, y_com_lo = compute_com_position(X_lo, p)
    vx_com_lo, vy_com_lo = compute_com_velocity(X_lo, p)

    if vy_com_lo <= 0:
        if verbose:
            print(f'Liftoff velocity negative: vy={vy_com_lo:.4f} m/s')
        return (0.0, t1, X1, None, None) if return_trajectories else 0.0

    if verbose:
        print(f'Liftoff at t={te1:.4f} s')
        print(f'  COM position: ({x_com_lo:.4f}, {y_com_lo:.4f}) m')
        print(f'  COM velocity: ({vx_com_lo:.4f}, {vy_com_lo:.4f}) m/s')

    # ── Phase 2 initial conditions ────────────────────────────────────────────
    # Zero angular velocities at liftoff for clean flight animation
    X0_p2 = np.array([
        x_com_lo, y_com_lo,
        X_lo[0], X_lo[1], X_lo[2],
        vx_com_lo, vy_com_lo,
        0.0, 0.0, 0.0
    ])

    # ── Phase 2: free-body flight ─────────────────────────────────────────────
    try:
        sol2 = solve_ivp(
            fun      = lambda t, X: jump_ode_phase2(t, X, p),
            t_span   = (te1, te1 + 2.0),
            y0       = X0_p2,
            method   = 'RK45',
            rtol     = 1e-8,
            atol     = 1e-10,
            max_step = 1e-3,
            dense_output = False
        )
    except Exception as e:
        if verbose:
            print(f'Phase 2 solver failed: {e}')
        return (0.0, t1, X1, None, None) if return_trajectories else 0.0

    t2 = sol2.t
    X2 = sol2.y.T   # shape (n_steps, 10)

    # ── Jump height ────────────────────────────────────────────────────────────
    jump_height = float(np.max(X2[:, 1]) - y_com_lo)

    if verbose:
        print(f'Jump height: {jump_height:.4f} m ({jump_height*100:.2f} cm)')
        print(f'Peak COM:    {np.max(X2[:,1]):.4f} m')
        print(f'Flight time: {t2[-1]-te1:.4f} s')

    if return_trajectories:
        return jump_height, t1, X1, t2, X2
    return jump_height


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Default parameters matching run_simulation.m baseline
    params_vec = np.array([
        -0.1,  0.6, -1.1,    # theta0 (initial angles)
         600., 1500., 1200.,  # tau_max (Nm)
         0.05,  0.04,  0.03,  # t_on (s)
         0.30,  0.30,  0.30   # t_dur (s)
    ])

    print('Running Python simulation...')
    h, t1, X1, t2, X2 = simulate_jump(
        params_vec,
        return_trajectories=True,
        verbose=True
    )
    print(f'\nFinal result: jump_height = {h:.4f} m ({h*100:.2f} cm)')
