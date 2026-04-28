"""
optimize.py
===========
Bayesian optimization of 3-segment jump model using BOTorch.

Maximizes jump height by tuning 12 parameters:
    theta0  [3] - initial joint angles (rad)
    tau_max [3] - peak torques (Nm)
    t_on    [3] - activation onset times (s)
    t_dur   [3] - activation durations (s)

Architecture:
    - Surrogate model: Gaussian Process (SingleTaskGP)
    - Acquisition function: Expected Improvement (LogEI)
    - Optimizer: L-BFGS-B on acquisition function
    - Warm start: Sobol quasi-random initial samples

Usage:
    python optimize.py

Results saved to: optimization_results.csv
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
from pathlib import Path

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

from gpytorch.mlls import ExactMarginalLogLikelihood

from simulator import simulate_jump, get_default_params

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER BOUNDS
# ─────────────────────────────────────────────────────────────────────────────
# 12 parameters in order:
#   [theta1, theta2, theta3, tau_ankle, tau_knee, tau_hip,
#    t_on_ankle, t_on_knee, t_on_hip,
#    t_dur_ankle, t_dur_knee, t_dur_hip]

BOUNDS = torch.tensor([
    # theta0 (initial angles, rad)
    [-1.50,  0.50, -2.50],   # lower
    [ 0.00,  2.50, -0.50],   # upper
    # tau_max (Nm)
    # t_on (s)
    # t_dur (s)
], dtype=torch.double)

# Build full bounds tensor
LOWER_BOUNDS = torch.tensor([
    -1.50,   # theta1
     0.50,   # theta2
    -2.50,   # theta3
    10.0,    # tau_max_ankle (Nm) — realistic ankle plantarflexion
    10.0,    # tau_max_knee
    10.0,    # tau_max_hip
    0.01,    # t_on_ankle (s)
    0.01,    # t_on_knee
    0.01,    # t_on_hip
    0.05,    # t_dur_ankle (s)
    0.05,    # t_dur_knee
    0.05,    # t_dur_hip
], dtype=torch.double)

UPPER_BOUNDS = torch.tensor([
     0.00,   # theta1
     2.50,   # theta2
    -0.50,   # theta3
    250.0,   # tau_max_ankle — scaled to segment inertias
    600.0,   # tau_max_knee
    500.0,   # tau_max_hip
     0.40,   # t_on_ankle
     0.40,   # t_on_knee
     0.40,   # t_on_hip
     0.60,   # t_dur_ankle
     0.60,   # t_dur_knee
     0.60,   # t_dur_hip
], dtype=torch.double)

BOUNDS_TENSOR = torch.stack([LOWER_BOUNDS, UPPER_BOUNDS])  # (2, 12)
DIM = 12   # number of optimization parameters


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def objective(params_vec_np, params):
    """
    Evaluate jump height for a given parameter vector.
    Returns 0.0 for infeasible/failed simulations.
    Clips negative values to 0 (can't have negative jump height).
    """
    try:
        h = simulate_jump(params_vec_np, params)
        return max(float(h), 0.0)
    except Exception as e:
        return 0.0


def evaluate_batch(X_batch, params, verbose=True):
    """
    Evaluate objective for a batch of candidate points.

    Parameters
    ----------
    X_batch : torch.Tensor, shape (n, 12)
        Candidate parameter vectors (normalized to [0,1] space).
    params : dict
        Model parameters.

    Returns
    -------
    Y : torch.Tensor, shape (n, 1)
        Jump heights in meters.
    """
    n = X_batch.shape[0]
    Y = torch.zeros(n, 1, dtype=torch.double)

    for i in range(n):
        # Unnormalize from [0,1] to physical units
        x_phys = unnormalize(X_batch[i])
        h = objective(x_phys.numpy(), params)
        Y[i, 0] = h
        if verbose:
            print(f'  [{i+1}/{n}] params={x_phys.numpy().round(3)} -> h={h*100:.2f} cm')

    return Y


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x):
    """Map physical parameter vector to [0,1]^12."""
    return (x - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)

def unnormalize(x):
    """Map [0,1]^12 back to physical parameter space."""
    return x * (UPPER_BOUNDS - LOWER_BOUNDS) + LOWER_BOUNDS


# ─────────────────────────────────────────────────────────────────────────────
#  GP MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_gp(train_X, train_Y):
    """
    Fit a SingleTaskGP surrogate model to observed data.

    Parameters
    ----------
    train_X : torch.Tensor, shape (n, 12)  — normalized inputs
    train_Y : torch.Tensor, shape (n, 1)   — observed jump heights

    Returns
    -------
    model : fitted GP model
    mll   : marginal log likelihood (for diagnostics)
    """
    model = SingleTaskGP(
        train_X, train_Y,
        outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model, mll


# ─────────────────────────────────────────────────────────────────────────────
#  ACQUISITION FUNCTION OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def get_next_candidate(model, train_Y, bounds_normalized):
    """
    Optimize the LogEI acquisition function to get the next candidate.

    Parameters
    ----------
    model : fitted GP
    train_Y : observed values (to compute best_f)
    bounds_normalized : (2, 12) tensor of [0,1] bounds

    Returns
    -------
    candidate : torch.Tensor, shape (1, 12) — normalized
    acq_value : float — acquisition function value at candidate
    """
    best_f = train_Y.max()

    acq_fn = LogExpectedImprovement(
        model   = model,
        best_f  = best_f,
    )

    candidate, acq_value = optimize_acqf(
        acq_function = acq_fn,
        bounds       = bounds_normalized,
        q            = 1,          # one point at a time
        num_restarts = 10,         # multi-start optimization
        raw_samples  = 256,        # random initial points for restarts
        options      = {'maxiter': 200}
    )

    return candidate, acq_value.item()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN OPTIMIZATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_optimization(
    n_initial  = 20,    # number of Sobol warm-start evaluations
    n_iter     = 50,    # number of BO iterations
    results_file = 'optimization_results.csv',
    verbose    = True
):
    """
    Run the full Bayesian optimization loop.

    Parameters
    ----------
    n_initial : int
        Number of quasi-random initial samples (Sobol sequence).
    n_iter : int
        Number of BO iterations after warm start.
    results_file : str
        CSV file to save all evaluated points.
    verbose : bool
        Print progress.

    Returns
    -------
    best_params : np.ndarray, shape (12,)
        Best parameter vector found (physical units).
    best_height : float
        Best jump height found (meters).
    history : pd.DataFrame
        Full optimization history.
    """
    print('=' * 65)
    print('  Bayesian Optimization of 3-Segment Jump Model')
    print('=' * 65)
    print(f'  Parameters: {DIM}')
    print(f'  Initial samples: {n_initial}')
    print(f'  BO iterations: {n_iter}')
    print(f'  Total evaluations: {n_initial + n_iter}')
    print('=' * 65)

    # Model parameters (fixed throughout optimization)
    params = get_default_params()

    # Normalized bounds for BOTorch [0,1]^12
    bounds_norm = torch.stack([
        torch.zeros(DIM, dtype=torch.double),
        torch.ones(DIM, dtype=torch.double)
    ])

    # Storage
    history = []
    t_start = time.time()

    # ── Phase 1: Sobol quasi-random warm start ────────────────────────────────
    print(f'\n--- Phase 1: Sobol warm start ({n_initial} evaluations) ---')

    sobol_X = draw_sobol_samples(
        bounds = bounds_norm,
        n      = n_initial,
        q      = 1
    ).squeeze(1)   # shape (n_initial, 12)

    # Also include the known-good baseline as one of the initial points
    baseline = normalize(torch.tensor([
        -0.1,  0.6, -1.1,       # theta0
         600., 1500., 1200.,    # tau_max
         0.05,  0.04,  0.03,    # t_on
         0.30,  0.30,  0.30     # t_dur
    ], dtype=torch.double))
    sobol_X[0] = baseline   # replace first Sobol point with baseline

    print(f'Evaluating {n_initial} initial points...')
    train_Y = evaluate_batch(sobol_X, params, verbose=verbose)
    train_X = sobol_X

    # Record initial evaluations
    for i in range(n_initial):
        x_phys = unnormalize(train_X[i]).numpy()
        history.append({
            'iteration': i - n_initial,   # negative = warm start
            'phase': 'sobol',
            'jump_height_m':  float(train_Y[i, 0]),
            'jump_height_cm': float(train_Y[i, 0]) * 100,
            'best_so_far_cm': float(train_Y[:i+1].max()) * 100,
            **{f'p{j+1}': x_phys[j] for j in range(DIM)}
        })

    best_idx = train_Y.argmax()
    print(f'\nBest after warm start: {float(train_Y.max())*100:.2f} cm')
    print(f'Best params: {unnormalize(train_X[best_idx]).numpy().round(3)}')

    # ── Phase 2: Bayesian optimization ───────────────────────────────────────
    print(f'\n--- Phase 2: Bayesian optimization ({n_iter} iterations) ---')

    for iteration in range(n_iter):
        iter_start = time.time()

        # Fit GP surrogate
        try:
            model, _ = fit_gp(train_X, train_Y)
        except Exception as e:
            print(f'  Warning: GP fitting failed at iteration {iteration}: {e}')
            print('  Falling back to random sample.')
            candidate = draw_sobol_samples(bounds_norm, n=1, q=1).squeeze(1)
            acq_val   = 0.0
        else:
            # Optimize acquisition function
            candidate, acq_val = get_next_candidate(model, train_Y, bounds_norm)

        # Evaluate objective at candidate
        x_phys = unnormalize(candidate.squeeze(0))
        h = objective(x_phys.numpy(), params)
        new_Y = torch.tensor([[h]], dtype=torch.double)

        # Update training data
        train_X = torch.cat([train_X, candidate.squeeze(1)], dim=0)
        train_Y = torch.cat([train_Y, new_Y],                dim=0)

        best_so_far = float(train_Y.max()) * 100
        iter_time   = time.time() - iter_start

        if verbose:
            print(f'  Iter {iteration+1:3d}/{n_iter} | '
                  f'h={h*100:6.2f} cm | '
                  f'best={best_so_far:6.2f} cm | '
                  f'acq={acq_val:.4f} | '
                  f't={iter_time:.1f}s')

        # Record
        history.append({
            'iteration':      iteration + 1,
            'phase':          'bo',
            'jump_height_m':  h,
            'jump_height_cm': h * 100,
            'best_so_far_cm': best_so_far,
            **{f'p{j+1}': float(x_phys[j]) for j in range(DIM)}
        })

    # ── Results ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    best_idx   = int(train_Y.argmax())
    best_X_phys = unnormalize(train_X[best_idx]).numpy()
    best_height = float(train_Y.max())

    print('\n' + '=' * 65)
    print('  OPTIMIZATION COMPLETE')
    print('=' * 65)
    print(f'  Total time:    {total_time:.1f} s ({total_time/60:.1f} min)')
    print(f'  Best height:   {best_height*100:.2f} cm')
    print(f'\n  Best parameters:')
    param_names = ['theta1','theta2','theta3',
                   'tau_ankle','tau_knee','tau_hip',
                   't_on_ankle','t_on_knee','t_on_hip',
                   't_dur_ankle','t_dur_knee','t_dur_hip']
    for name, val in zip(param_names, best_X_phys):
        print(f'    {name:<15} = {val:.4f}')

    # Save history
    df = pd.DataFrame(history)
    df.to_csv(results_file, index=False)
    print(f'\n  Results saved to: {results_file}')

    # Plot convergence
    plot_convergence(df, best_height)

    return best_X_phys, best_height, df


# ─────────────────────────────────────────────────────────────────────────────
#  CONVERGENCE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(df, best_height):
    """Plot optimization convergence curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('BOTorch Optimization Convergence', fontsize=13, fontweight='bold')

    bo_df = df[df['phase'] == 'bo']
    all_df = df.copy()
    all_df['eval_number'] = range(len(all_df))

    # ── Left: all evaluations ─────────────────────────────────────────────────
    ax1 = axes[0]
    sobol_df = all_df[all_df['phase'] == 'sobol']
    bo_plot  = all_df[all_df['phase'] == 'bo']

    ax1.scatter(sobol_df['eval_number'], sobol_df['jump_height_cm'],
                color='gray', alpha=0.5, s=30, label='Sobol init', zorder=2)
    ax1.scatter(bo_plot['eval_number'], bo_plot['jump_height_cm'],
                color='steelblue', alpha=0.7, s=30, label='BO eval', zorder=2)
    ax1.plot(all_df['eval_number'], all_df['best_so_far_cm'],
             color='red', lw=2, label='Best so far', zorder=3)
    ax1.axhline(best_height*100, color='red', ls='--', lw=1.5, alpha=0.5)
    ax1.set_xlabel('Evaluation number')
    ax1.set_ylabel('Jump height (cm)')
    ax1.set_title('All Evaluations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right: BO phase only (convergence) ───────────────────────────────────
    ax2 = axes[1]
    if len(bo_df) > 0:
        ax2.plot(bo_df['iteration'], bo_df['best_so_far_cm'],
                 color='red', lw=2.5, marker='o', markersize=4, label='Best so far')
        ax2.fill_between(bo_df['iteration'],
                         bo_df['best_so_far_cm'].min(),
                         bo_df['best_so_far_cm'],
                         alpha=0.15, color='red')
    ax2.set_xlabel('BO iteration')
    ax2.set_ylabel('Best jump height (cm)')
    ax2.set_title('BO Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimization_convergence.png', dpi=150, bbox_inches='tight')
    print('  Convergence plot saved to: optimization_convergence.png')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    best_params, best_height, history = run_optimization(
        n_initial = 20,    # Sobol warm-start evaluations
        n_iter    = 50,    # BO iterations
        verbose   = True
    )

    print('\n--- Running final simulation with best parameters ---')
    from simulator import simulate_jump
    h, t1, X1, t2, X2 = simulate_jump(
        best_params,
        return_trajectories=True,
        verbose=True
    )
