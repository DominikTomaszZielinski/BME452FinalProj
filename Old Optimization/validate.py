"""
validate.py
===========
Side-by-side validation of Python simulator against MATLAB implementation.

Runs the same parameters through both simulators and compares:
  - COM trajectory (x, y vs time)
  - Joint angles (theta1, theta2, theta3 vs time)
  - Liftoff time, velocity, and jump height
  - Phase 1 torque profiles

Usage:
    python validate.py

Requirements:
    - MATLAB must be open and matlab.engine.shareEngine called in MATLAB
    - MATLAB files must be on the MATLAB path
    - simulator.py must be in the same directory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matlab.engine
import time
from simulator import (
    simulate_jump,
    get_default_params,
    compute_com_position,
    compute_com_velocity,
    compute_torques,
    jump_ode_phase1,
    jump_ode_phase2,
    make_liftoff_event
)
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETERS  (must match run_simulation.m exactly)
# ─────────────────────────────────────────────────────────────────────────────

PARAMS_VEC = np.array([
    -0.1,  0.6, -1.1,      # theta0 (initial angles, rad)
     600., 1500., 1200.,   # tau_max (Nm)
     0.05,  0.04,  0.03,   # t_on (s)
     0.30,  0.30,  0.30    # t_dur (s)
])

MATLAB_SCRIPT_PATH = r'C:\Users\This PRC\Documents\BME452FinalProj'


# ─────────────────────────────────────────────────────────────────────────────
#  RUN PYTHON SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_python(params_vec, verbose=True):
    """Run Python simulation and return all trajectory data."""
    p = get_default_params()

    theta0  = params_vec[0:3]
    tau_max = params_vec[3:6]
    t_on    = params_vec[6:9]
    t_dur   = params_vec[9:12]

    ctrl = {'tau_max': tau_max, 't_on': t_on, 't_dur': t_dur, 'k': 50.0}
    X0_p1 = np.concatenate([theta0, np.zeros(3)])

    # Phase 1
    liftoff_ev = make_liftoff_event(p, ctrl)
    sol1 = solve_ivp(
        fun      = lambda t, X: jump_ode_phase1(t, X, p, ctrl),
        t_span   = (0.0, 1.0),
        y0       = X0_p1,
        method   = 'Radau',
        rtol     = 1e-6, atol = 1e-8, max_step = 1e-4,
        events   = liftoff_ev
    )

    t1  = sol1.t
    X1  = sol1.y.T
    te1 = sol1.t_events[0][0]

    X_lo = X1[-1]
    x_com_lo, y_com_lo   = compute_com_position(X_lo, p)
    vx_com_lo, vy_com_lo = compute_com_velocity(X_lo, p)

    # Phase 2 (angular velocities zeroed at liftoff)
    X0_p2 = np.array([
        x_com_lo, y_com_lo,
        X_lo[0], X_lo[1], X_lo[2],
        vx_com_lo, vy_com_lo,
        0.0, 0.0, 0.0
    ])
    sol2 = solve_ivp(
        fun      = lambda t, X: jump_ode_phase2(t, X, p),
        t_span   = (te1, te1 + 2.0),
        y0       = X0_p2,
        method   = 'RK45',
        rtol     = 1e-8, atol = 1e-10, max_step = 1e-3
    )
    t2 = sol2.t
    X2 = sol2.y.T

    jump_height = float(np.max(X2[:, 1]) - y_com_lo)

    # Torque profiles over phase 1
    tau_hist = np.array([compute_torques(t, ctrl) for t in t1])

    if verbose:
        print(f'\n=== Python Simulation ===')
        print(f'Liftoff at t = {te1:.4f} s')
        print(f'COM at liftoff: ({x_com_lo:.4f}, {y_com_lo:.4f}) m')
        print(f'Velocity at liftoff: ({vx_com_lo:.4f}, {vy_com_lo:.4f}) m/s')
        print(f'Jump height: {jump_height:.4f} m ({jump_height*100:.2f} cm)')

    return {
        't1': t1, 'X1': X1, 'tau_hist': tau_hist,
        't2': t2, 'X2': X2,
        'te1': te1,
        'x_com_lo': x_com_lo, 'y_com_lo': y_com_lo,
        'vx_com_lo': vx_com_lo, 'vy_com_lo': vy_com_lo,
        'jump_height': jump_height,
        'p': p, 'ctrl': ctrl
    }


# ─────────────────────────────────────────────────────────────────────────────
#  RUN MATLAB SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_matlab(params_vec, eng, verbose=True):
    """
    Run MATLAB simulation via matlab.engine and extract trajectory data.
    Calls simulate_jump_export() which must exist in run_simulation.m
    """
    print('\nConnecting to MATLAB...')

    # Add MATLAB project path
    eng.addpath(MATLAB_SCRIPT_PATH, nargout=0)

    # Convert params to MATLAB double array
    pv = matlab.double(params_vec.tolist())

    print('Running MATLAB simulation...')
    t0 = time.time()

    try:
        # Call the MATLAB wrapper function
        result = eng.simulate_jump_export(pv, nargout=8)
        elapsed = time.time() - t0
        print(f'MATLAB simulation completed in {elapsed:.2f}s')

        # Unpack results
        t1_ml   = np.array(result[0]).flatten()
        X1_ml   = np.array(result[1])           # (n x 6)
        tau_ml  = np.array(result[2])           # (n x 3)
        te1_ml  = float(result[3])
        t2_ml   = np.array(result[4]).flatten()
        X2_ml   = np.array(result[5])           # (n x 10)
        jh_ml   = float(result[6])
        lo_ml   = np.array(result[7]).flatten() # [x_lo, y_lo, vx_lo, vy_lo]

        if verbose:
            print(f'\n=== MATLAB Simulation ===')
            print(f'Liftoff at t = {te1_ml:.4f} s')
            print(f'COM at liftoff: ({lo_ml[0]:.4f}, {lo_ml[1]:.4f}) m')
            print(f'Velocity at liftoff: ({lo_ml[2]:.4f}, {lo_ml[3]:.4f}) m/s')
            print(f'Jump height: {jh_ml:.4f} m ({jh_ml*100:.2f} cm)')

        return {
            't1': t1_ml, 'X1': X1_ml, 'tau_hist': tau_ml,
            't2': t2_ml, 'X2': X2_ml,
            'te1': te1_ml,
            'x_com_lo': lo_ml[0], 'y_com_lo': lo_ml[1],
            'vx_com_lo': lo_ml[2], 'vy_com_lo': lo_ml[3],
            'jump_height': jh_ml
        }

    except Exception as e:
        print(f'MATLAB call failed: {e}')
        print('Make sure simulate_jump_export() exists in your MATLAB files.')
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(py, ml):
    """
    Generate side-by-side comparison plots of Python vs MATLAB results.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Python vs MATLAB Simulation Validation', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    col_py = '#1f77b4'   # blue  - Python
    col_ml = '#d62728'   # red   - MATLAB
    lw = 2.0

    # ── 1. COM trajectory ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])

    # Python COM position from phase 1
    com_py_x = []
    com_py_y = []
    p = py['p']
    for Xi in py['X1']:
        xc, yc = compute_com_position(Xi, p)
        com_py_x.append(xc)
        com_py_y.append(yc)
    com_py_x = np.array(com_py_x)
    com_py_y = np.array(com_py_y)

    # Phase 2 COM from state directly
    ax1.plot(com_py_x, com_py_y,
             color=col_py, lw=lw, label='Python - push-off')
    ax1.plot(py['X2'][:, 0], py['X2'][:, 1],
             color=col_py, lw=lw, ls='--', label='Python - flight')

    if ml is not None:
        # MATLAB phase 1 COM (approximate from angles)
        ax1.plot(ml['X2'][:, 0], ml['X2'][:, 1],
                 color=col_ml, lw=lw, ls='--', label='MATLAB - flight')

    ax1.axhline(0, color='k', lw=1.5)
    ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)')
    ax1.set_title('COM Trajectory')
    ax1.legend(fontsize=8); ax1.grid(True)

    # ── 2. COM y vs time ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])

    # Phase 1 COM y
    t_p1 = py['t1']
    ax2.plot(t_p1, com_py_y, color=col_py, lw=lw, label='Python P1')
    # Phase 2 COM y
    ax2.plot(py['t2'], py['X2'][:, 1], color=col_py, lw=lw,
             ls='--', label='Python P2')

    if ml is not None:
        ax2.plot(ml['t2'], ml['X2'][:, 1], color=col_ml, lw=lw,
                 ls='--', label='MATLAB P2')

    ax2.axvline(py['te1'], color='gray', ls=':', lw=1.5, label='Liftoff')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('y_com (m)')
    ax2.set_title('COM Height vs Time')
    ax2.legend(fontsize=8); ax2.grid(True)

    # ── 3. Joint angles phase 1 ──────────────────────────────────────────────
    labels = [r'$\theta_1$ (foot)', r'$\theta_2$ (shank)', r'$\theta_3$ (thigh)']
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(py['t1'], np.degrees(py['X1'][:, i]),
                color=col_py, lw=lw, label='Python')
        if ml is not None:
            ax.plot(ml['t1'], np.degrees(ml['X1'][:, i]),
                    color=col_ml, lw=lw, ls='--', label='MATLAB')
        ax.axvline(py['te1'], color='gray', ls=':', lw=1.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Angle (deg)')
        ax.set_title(labels[i])
        ax.legend(fontsize=8); ax.grid(True)

    # ── 4. Torque profiles ───────────────────────────────────────────────────
    torque_labels = ['Ankle', 'Knee', 'Hip']
    torque_colors = ['#2ca02c', '#9467bd', '#8c564b']
    ax_tau = fig.add_subplot(gs[2, :2])
    for i in range(3):
        ax_tau.plot(py['t1'], py['tau_hist'][:, i],
                    color=torque_colors[i], lw=lw,
                    label=f'{torque_labels[i]} (Py)')
        if ml is not None and ml['tau_hist'] is not None:
            ax_tau.plot(ml['t1'], ml['tau_hist'][:, i],
                        color=torque_colors[i], lw=lw, ls='--',
                        label=f'{torque_labels[i]} (ML)')
    ax_tau.axvline(py['te1'], color='gray', ls=':', lw=1.5, label='Liftoff')
    ax_tau.set_xlabel('Time (s)'); ax_tau.set_ylabel('Torque (Nm)')
    ax_tau.set_title('Joint Torque Profiles')
    ax_tau.legend(fontsize=7, ncol=2); ax_tau.grid(True)

    # ── 5. Summary table ─────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.axis('off')

    ml_jh  = f"{ml['jump_height']*100:.2f}" if ml else 'N/A'
    ml_te1 = f"{ml['te1']:.4f}"             if ml else 'N/A'
    ml_vy  = f"{ml['vy_com_lo']:.4f}"       if ml else 'N/A'

    err_jh  = abs(py['jump_height'] - ml['jump_height'])*100 if ml else float('nan')
    err_te1 = abs(py['te1'] - ml['te1'])*1000                if ml else float('nan')
    err_vy  = abs(py['vy_com_lo'] - ml['vy_com_lo'])         if ml else float('nan')

    table_data = [
        ['Metric',        'Python',                          'MATLAB',  'Diff'],
        ['Jump height',   f"{py['jump_height']*100:.2f} cm", f"{ml_jh} cm",
         f"{err_jh:.3f} cm"],
        ['Liftoff time',  f"{py['te1']:.4f} s",              f"{ml_te1} s",
         f"{err_te1:.3f} ms"],
        ['vy at liftoff', f"{py['vy_com_lo']:.4f} m/s",     f"{ml_vy} m/s",
         f"{err_vy:.4f} m/s"],
    ]

    tbl = ax_tbl.table(
        cellText  = table_data[1:],
        colLabels = table_data[0],
        loc       = 'center',
        cellLoc   = 'center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    ax_tbl.set_title('Validation Summary', fontsize=10, fontweight='bold')

    plt.savefig('validation_comparison.png', dpi=150, bbox_inches='tight')
    print('\nValidation plot saved to: validation_comparison.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('=' * 60)
    print('  Jump Simulation Validation: Python vs MATLAB')
    print('=' * 60)

    # ── Step 1: Run Python simulation ─────────────────────────────────────────
    py_results = run_python(PARAMS_VEC, verbose=True)

    # ── Step 2: Connect to MATLAB and run ─────────────────────────────────────
    ml_results = None
    try:
        print('\nConnecting to shared MATLAB session...')
        print('(Make sure MATLAB is open and matlab.engine.shareEngine was called)')
        eng = matlab.engine.connect_matlab()
        print('Connected!')
        ml_results = run_matlab(PARAMS_VEC, eng, verbose=True)
        eng.quit()
    except Exception as e:
        print(f'\nCould not connect to MATLAB: {e}')
        print('Plotting Python results only.')

    # ── Step 3: Print comparison ───────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  VALIDATION SUMMARY')
    print('=' * 60)
    print(f"{'Metric':<25} {'Python':>15} {'MATLAB':>15} {'Difference':>15}")
    print('-' * 70)

    metrics = [
        ('Jump height (cm)',    py_results['jump_height']*100,
         ml_results['jump_height']*100 if ml_results else None),
        ('Liftoff time (s)',    py_results['te1'],
         ml_results['te1'] if ml_results else None),
        ('vy at liftoff (m/s)',py_results['vy_com_lo'],
         ml_results['vy_com_lo'] if ml_results else None),
        ('x_com liftoff (m)',  py_results['x_com_lo'],
         ml_results['x_com_lo'] if ml_results else None),
        ('y_com liftoff (m)',  py_results['y_com_lo'],
         ml_results['y_com_lo'] if ml_results else None),
    ]

    for name, py_val, ml_val in metrics:
        if ml_val is not None:
            diff = abs(py_val - ml_val)
            pct  = abs(diff / ml_val * 100) if ml_val != 0 else 0
            print(f"{name:<25} {py_val:>15.4f} {ml_val:>15.4f} "
                  f"{diff:>10.4f} ({pct:.2f}%)")
        else:
            print(f"{name:<25} {py_val:>15.4f} {'N/A':>15}")

    # ── Step 4: Plot ───────────────────────────────────────────────────────────
    plot_comparison(py_results, ml_results)
