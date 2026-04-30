"""
optimize.py
===========
Multi-method optimization of the 3-segment jump model.

Optimization target: maximize jump height by tuning starting joint angles only.
    params = [theta1, theta2, theta3]   (3 parameters)

Torques are fixed at literature-based maxima (step function, t=0 to liftoff):
    ankle : 175 N·m  (Hussain & Frey-Law 2016)
    knee  : 247 N·m  (Harbo et al. 2012)
    hip   : 175 N·m  (Harbo et al. 2012)

Three optimization methods are compared:
    1. Bayesian Optimization (BOTorch, LogEI acquisition)
       — Sample-efficient, builds a GP surrogate to guide search.
         Well-suited for expensive black-box objectives with few evaluations.
    2. Differential Evolution (scipy.optimize.differential_evolution)
       — Population-based global optimizer; robust to multimodal landscapes.
         No gradient information needed; good for discontinuous objectives.
    3. CMA-ES (pycma)
       — Covariance Matrix Adaptation Evolution Strategy; a gradient-free
         method that adapts its search distribution to the landscape.
         Often the strongest single-start optimizer for smooth landscapes.

Three noise rounds are run for each method:
    Round 1 (clean)   : no noise added to parameters
    Round 2 (low noise) : Gaussian noise, σ = 0.02 rad on each angle
    Round 3 (high noise): Gaussian noise, σ = 0.05 rad on each angle

Results are collected in a single comparison plot showing best height found
by each method × noise level as grouped bars, plus convergence curves.

Usage:
    python optimize.py

Requirements:
    pip install botorch gpytorch scipy cma
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings
from pathlib import Path

# BOTorch imports (optional — graceful fallback if torch/botorch not installed)
BOTORCH_AVAILABLE = False
try:
    import torch as _torch_test
    BOTORCH_AVAILABLE = True
except (ImportError, OSError):
    pass

if BOTORCH_AVAILABLE:
    try:
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import LogExpectedImprovement
        from botorch.optim import optimize_acqf
        from botorch.utils.sampling import draw_sobol_samples
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except (ImportError, OSError):
        BOTORCH_AVAILABLE = False

if not BOTORCH_AVAILABLE:
    print('[WARNING] BOTorch/torch not available — BO will use random Sobol-like search.')

# scipy DE
from scipy.optimize import differential_evolution

# CMA-ES
import cma

from simulator import simulate_jump, get_default_params, TAU_MAX_ANKLE, TAU_MAX_KNEE, TAU_MAX_HIP

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER BOUNDS  (3 parameters: starting joint angles only)
# ─────────────────────────────────────────────────────────────────────────────
#
#  theta1 : ankle absolute angle (foot) — small negative to slightly positive
#  theta2 : knee relative angle (shank rel. foot) — positive (flexed)
#  theta3 : hip relative angle (thigh rel. shank) — negative (flexed)

LOWER_BOUNDS = np.array([-0.60,  0.20, -2.20])   # rad
UPPER_BOUNDS = np.array([ 0.10,  1.80, -0.30])   # rad

DIM = 3

NOISE_LEVELS = {
    'clean':      0.00,
    'low_noise':  0.02,
    'high_noise': 0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
#  OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def objective(theta_vec, p, noise_std=0.0):
    """
    Evaluate jump height for a given starting-angle vector.
    If noise_std > 0, adds Gaussian noise to each angle before simulating,
    then averages over N_NOISE_TRIALS trials to get a robust estimate.

    Parameters
    ----------
    theta_vec : array-like, shape (3,)
    p : dict  — model parameters
    noise_std : float  — std dev of angle perturbation in radians

    Returns
    -------
    float — jump height in meters (≥ 0)
    """
    theta_vec = np.asarray(theta_vec, dtype=float)

    if noise_std > 0.0:
        N_TRIALS = 5
        heights = []
        for _ in range(N_TRIALS):
            noisy = theta_vec + np.random.randn(DIM) * noise_std
            noisy = np.clip(noisy, LOWER_BOUNDS, UPPER_BOUNDS)
            try:
                h = simulate_jump(noisy, p)
                heights.append(max(float(h), 0.0))
            except Exception:
                heights.append(0.0)
        return float(np.mean(heights))
    else:
        try:
            h = simulate_jump(theta_vec, p)
            return max(float(h), 0.0)
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALIZATION HELPERS  (for BOTorch, which works in [0,1]^d)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x_np):
    return (np.asarray(x_np) - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)

def unnormalize(x_np):
    return np.asarray(x_np) * (UPPER_BOUNDS - LOWER_BOUNDS) + LOWER_BOUNDS


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 1 — BAYESIAN OPTIMIZATION  (BOTorch + SingleTaskGP + LogEI)
# ─────────────────────────────────────────────────────────────────────────────

def run_bayesian_optimization(p, noise_std=0.0,
                               n_initial=12, n_iter=40, verbose=True):
    """
    Bayesian optimization over 3-dimensional starting-angle space.

    Uses a Gaussian Process surrogate (SingleTaskGP) with Log Expected
    Improvement as the acquisition function, seeded by a Sobol sequence.

    Parameters
    ----------
    p : dict — model parameters
    noise_std : float — angle perturbation std dev (0 = clean)
    n_initial : int — number of Sobol warm-start evaluations
    n_iter : int — number of BO iterations
    verbose : bool

    Returns
    -------
    best_theta : np.ndarray, shape (3,)
    best_height : float  (meters)
    convergence : list of float — best height found at each iteration
    """
    label = f'BO (noise={noise_std:.2f})'
    if verbose:
        print(f'\n  [{label}] Starting: {n_initial} init + {n_iter} BO iters')

    if not BOTORCH_AVAILABLE:
        # Fallback: pure quasi-random Sobol search (no GP surrogate)
        print('    [BO fallback] BOTorch unavailable — running Sobol random search.')
        rng       = np.random.RandomState(42)
        n_total   = n_initial + n_iter
        # Use scrambled Sobol-like samples via rng (simple uniform fallback)
        best_theta  = None
        best_height = 0.0
        convergence = []
        for i in range(n_total):
            theta = LOWER_BOUNDS + rng.rand(DIM) * (UPPER_BOUNDS - LOWER_BOUNDS)
            h     = objective(theta, p, noise_std)
            if h > best_height:
                best_height = h
                best_theta  = theta.copy()
            convergence.append(best_height * 100)
            if verbose and (i+1) % 10 == 0:
                print(f'    Sobol [{i+1:2d}/{n_total}] best={best_height*100:.2f} cm')
        if verbose:
            print(f'  [{label}] Done (fallback). Best = {best_height*100:.2f} cm')
        return best_theta, best_height, convergence

    # ── Full BOTorch path ─────────────────────────────────────────────────────
    bounds_norm = torch.stack([
        torch.zeros(DIM, dtype=torch.double),
        torch.ones(DIM, dtype=torch.double)
    ])

    # ── Sobol warm start ──────────────────────────────────────────────────────
    sobol_X = draw_sobol_samples(bounds_norm, n=n_initial, q=1).squeeze(1)

    train_Y_list = []
    for i in range(n_initial):
        theta = unnormalize(sobol_X[i].numpy())
        h = objective(theta, p, noise_std)
        train_Y_list.append(h)
        if verbose:
            print(f'    init [{i+1:2d}/{n_initial}] θ={theta.round(3)} h={h*100:.2f} cm')

    train_X = sobol_X
    train_Y = torch.tensor(train_Y_list, dtype=torch.double).unsqueeze(1)

    convergence = [float(train_Y.max()) * 100]

    # ── BO loop ───────────────────────────────────────────────────────────────
    for it in range(n_iter):
        try:
            model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll   = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq_fn = LogExpectedImprovement(model=model, best_f=train_Y.max())
            candidate, _ = optimize_acqf(
                acq_function = acq_fn,
                bounds       = bounds_norm,
                q            = 1,
                num_restarts = 8,
                raw_samples  = 128,
            )
        except Exception:
            candidate = draw_sobol_samples(bounds_norm, n=1, q=1).squeeze(1)

        theta = unnormalize(candidate.squeeze(0).detach().numpy())
        h = objective(theta, p, noise_std)

        train_X = torch.cat([train_X, candidate.squeeze(1)], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[h]], dtype=torch.double)], dim=0)

        best_now = float(train_Y.max()) * 100
        convergence.append(best_now)

        if verbose:
            print(f'    BO  [{it+1:2d}/{n_iter}] θ={theta.round(3)} '
                  f'h={h*100:.2f} cm | best={best_now:.2f} cm')

    best_idx    = int(train_Y.argmax())
    best_theta  = unnormalize(train_X[best_idx].numpy())
    best_height = float(train_Y.max())

    if verbose:
        print(f'  [{label}] Done. Best = {best_height*100:.2f} cm  '
              f'θ={best_theta.round(3)}')

    return best_theta, best_height, convergence


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 2 — DIFFERENTIAL EVOLUTION  (scipy)
# ─────────────────────────────────────────────────────────────────────────────

def run_differential_evolution(p, noise_std=0.0,
                                maxiter=40, popsize=6, verbose=True):
    """
    Differential Evolution over the 3-dimensional starting-angle space.

    DE is a population-based stochastic global optimizer that requires no
    gradient information and handles multimodal and discontinuous objectives
    well.  scipy's implementation uses a (popsize * DIM)-member population.

    Parameters
    ----------
    p : dict
    noise_std : float
    maxiter : int — maximum number of generations
    popsize : int — population multiplier (total pop = popsize * DIM)
    verbose : bool

    Returns
    -------
    best_theta : np.ndarray
    best_height : float
    convergence : list of float
    """
    label = f'DE (noise={noise_std:.2f})'
    if verbose:
        print(f'\n  [{label}] Starting: maxiter={maxiter}, pop={popsize*DIM}')

    convergence = []
    best_so_far = [0.0]
    call_count  = [0]

    def neg_objective(theta):
        """scipy minimize → negate for maximization."""
        h = objective(theta, p, noise_std)
        call_count[0] += 1
        if h > best_so_far[0]:
            best_so_far[0] = h
        convergence.append(best_so_far[0] * 100)
        if verbose and call_count[0] % 5 == 0:
            print(f'    DE  [eval {call_count[0]:4d}] h={h*100:.2f} cm | '
                  f'best={best_so_far[0]*100:.2f} cm')
        return -h

    bounds = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))

    result = differential_evolution(
        neg_objective,
        bounds,
        maxiter   = maxiter,
        popsize   = popsize,
        seed      = 42,
        tol       = 1e-5,
        mutation  = (0.5, 1.0),
        recombination = 0.7,
        strategy  = 'best1bin',
        polish    = False,     # L-BFGS-B polish disabled (can hang on non-smooth objectives)
    )

    best_theta  = result.x
    best_height = -result.fun

    if verbose:
        print(f'  [{label}] Done. Best = {best_height*100:.2f} cm  '
              f'θ={best_theta.round(3)}')

    return best_theta, best_height, convergence


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 3 — CMA-ES  (pycma)
# ─────────────────────────────────────────────────────────────────────────────

def run_cmaes(p, noise_std=0.0,
              maxiter=40, sigma0=0.3, verbose=True):
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) over the
    3-dimensional starting-angle space.

    CMA-ES adapts a full covariance matrix of a multivariate Gaussian search
    distribution.  It is often the strongest gradient-free optimizer for
    smooth continuous landscapes and handles ill-conditioned problems well.

    Parameters
    ----------
    p : dict
    noise_std : float
    maxiter : int — maximum generations
    sigma0 : float — initial step size in normalized [0,1] space
    verbose : bool

    Returns
    -------
    best_theta : np.ndarray
    best_height : float
    convergence : list of float
    """
    label = f'CMA-ES (noise={noise_std:.2f})'
    if verbose:
        print(f'\n  [{label}] Starting: maxiter={maxiter}, σ0={sigma0}')

    # CMA-ES works in normalized [0,1]^3 space
    x0_norm = normalize(np.array([-0.15, 0.80, -1.20]))   # reasonable starting guess

    convergence   = []
    best_so_far   = 0.0
    call_count    = 0

    def neg_objective_norm(x_norm):
        nonlocal best_so_far, call_count
        x_norm = np.clip(np.asarray(x_norm), 0.0, 1.0)
        theta  = unnormalize(x_norm)
        h = objective(theta, p, noise_std)
        call_count += 1
        if h > best_so_far:
            best_so_far = h
        convergence.append(best_so_far * 100)
        if verbose and call_count % 5 == 0:
            print(f'    CMA [eval {call_count:4d}] h={h*100:.2f} cm | '
                  f'best={best_so_far*100:.2f} cm')
        return -h

    opts = cma.CMAOptions()
    opts['maxiter']      = maxiter
    opts['bounds']       = [[0.0]*DIM, [1.0]*DIM]
    opts['verbose']      = -9         # suppress CMA's own output
    opts['tolx']         = 1e-4
    opts['tolfun']       = 1e-5
    opts['seed']         = 42

    es = cma.CMAEvolutionStrategy(x0_norm, sigma0, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [neg_objective_norm(x) for x in solutions]
        es.tell(solutions, fitnesses)

    res = es.result
    best_x_norm = np.clip(res.xbest, 0.0, 1.0)
    best_theta  = unnormalize(best_x_norm)
    best_height = -res.fbest

    if verbose:
        print(f'  [{label}] Done. Best = {best_height*100:.2f} cm  '
              f'θ={best_theta.round(3)}')

    return best_theta, best_height, convergence


# ─────────────────────────────────────────────────────────────────────────────
#  RUNNER  —  ALL METHODS × ALL NOISE ROUNDS
# ─────────────────────────────────────────────────────────────────────────────

def run_all(n_initial_bo=12, n_iter_bo=40,
            maxiter_de=40,  popsize_de=6,
            maxiter_cma=40, sigma_cma=0.3,
            verbose=True):
    """
    Run all 3 optimizers × 3 noise levels = 9 total experiments.

    Returns
    -------
    results : dict
        results[method][noise_label] = {
            'best_theta'  : np.ndarray (3,)
            'best_height' : float (m)
            'convergence' : list of float (cm), length = number of evaluations
        }
    """
    p = get_default_params()
    results = {m: {} for m in ('BO', 'DE', 'CMA-ES')}

    for noise_label, noise_std in NOISE_LEVELS.items():
        print('\n' + '=' * 65)
        print(f'  NOISE ROUND: {noise_label}  (σ = {noise_std:.2f} rad)')
        print('=' * 65)

        # --- Bayesian Optimization ---
        theta, h, conv = run_bayesian_optimization(
            p, noise_std,
            n_initial=n_initial_bo, n_iter=n_iter_bo,
            verbose=verbose
        )
        results['BO'][noise_label] = {
            'best_theta': theta, 'best_height': h, 'convergence': conv
        }

        # --- Differential Evolution ---
        theta, h, conv = run_differential_evolution(
            p, noise_std,
            maxiter=maxiter_de, popsize=popsize_de,
            verbose=verbose
        )
        results['DE'][noise_label] = {
            'best_theta': theta, 'best_height': h, 'convergence': conv
        }

        # --- CMA-ES ---
        theta, h, conv = run_cmaes(
            p, noise_std,
            maxiter=maxiter_cma, sigma0=sigma_cma,
            verbose=verbose
        )
        results['CMA-ES'][noise_label] = {
            'best_theta': theta, 'best_height': h, 'convergence': conv
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results):
    methods     = ['BO', 'DE', 'CMA-ES']
    noise_labs  = list(NOISE_LEVELS.keys())
    param_names = ['θ1 (ankle)', 'θ2 (knee)', 'θ3 (hip)']

    print('\n' + '=' * 65)
    print('  RESULTS SUMMARY')
    print('=' * 65)
    print(f'  Fixed torques: ankle={TAU_MAX_ANKLE} N·m, '
          f'knee={TAU_MAX_KNEE} N·m, hip={TAU_MAX_HIP} N·m')
    print()

    # Header
    col_w = 14
    header = 'Method'.ljust(10) + 'Noise'.ljust(14)
    header += 'Height (cm)'.ljust(col_w)
    for n in param_names:
        header += n.ljust(col_w)
    print(header)
    print('-' * (10 + 14 + col_w * (1 + len(param_names))))

    for method in methods:
        for nl in noise_labs:
            r = results[method][nl]
            row  = method.ljust(10) + nl.ljust(14)
            row += f'{r["best_height"]*100:.2f}'.ljust(col_w)
            for v in r['best_theta']:
                row += f'{v:.4f}'.ljust(col_w)
            print(row)
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'BO':     '#2196F3',   # blue
    'DE':     '#4CAF50',   # green
    'CMA-ES': '#FF5722',   # orange-red
}

NOISE_STYLES = {
    'clean':      ('-',  'o', 1.0),
    'low_noise':  ('--', 's', 0.75),
    'high_noise': (':',  '^', 0.55),
}

def plot_results(results, save_path='optimization_comparison.png'):
    """
    Two-panel figure:
      Left  — grouped bar chart of best height per method × noise level
      Right — convergence curves for all 9 combinations
    """
    methods     = ['BO', 'DE', 'CMA-ES']
    noise_labs  = list(NOISE_LEVELS.keys())
    noise_sigma = list(NOISE_LEVELS.values())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        '3-Segment Jump Optimization: Method × Noise Comparison\n'
        f'Fixed torques — ankle: {TAU_MAX_ANKLE} N·m | '
        f'knee: {TAU_MAX_KNEE} N·m | hip: {TAU_MAX_HIP} N·m',
        fontsize=12, fontweight='bold', y=1.02
    )

    # ── Left: grouped bar chart ────────────────────────────────────────────
    ax1 = axes[0]
    n_methods = len(methods)
    n_noise   = len(noise_labs)
    bar_w     = 0.22
    group_gap = 1.0

    group_positions = np.arange(n_noise) * group_gap
    offsets = np.linspace(-(n_methods-1)/2, (n_methods-1)/2, n_methods) * bar_w

    for mi, method in enumerate(methods):
        heights_cm = [results[method][nl]['best_height'] * 100 for nl in noise_labs]
        x_pos = group_positions + offsets[mi]
        bars = ax1.bar(
            x_pos, heights_cm,
            width  = bar_w,
            color  = COLORS[method],
            alpha  = 0.85,
            label  = method,
            zorder = 3,
            edgecolor = 'white',
            linewidth = 0.5,
        )
        for bar, hcm in zip(bars, heights_cm):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{hcm:.1f}',
                ha='center', va='bottom',
                fontsize=7.5, fontweight='bold',
                color=COLORS[method]
            )

    ax1.set_xticks(group_positions)
    ax1.set_xticklabels([
        f'{nl}\n(σ={s:.2f} rad)'
        for nl, s in zip(noise_labs, noise_sigma)
    ], fontsize=9)
    ax1.set_ylabel('Best Jump Height (cm)', fontsize=10)
    ax1.set_title('Best Height by Method & Noise Level', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3, zorder=0)
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.12)

    # ── Right: convergence curves ──────────────────────────────────────────
    ax2 = axes[1]

    for method in methods:
        for nl in noise_labs:
            ls, mk, alpha = NOISE_STYLES[nl]
            conv = results[method][nl]['convergence']
            evals = np.arange(len(conv))
            ax2.plot(
                evals, conv,
                color     = COLORS[method],
                linestyle = ls,
                marker    = mk,
                markevery = max(1, len(conv)//8),
                markersize = 4,
                linewidth  = 1.6,
                alpha      = alpha,
                label      = f'{method} | {nl}',
            )

    ax2.set_xlabel('Cumulative Function Evaluations', fontsize=10)
    ax2.set_ylabel('Best Jump Height (cm)', fontsize=10)
    ax2.set_title('Convergence Curves — All Methods × Noise Levels', fontsize=10)
    ax2.legend(fontsize=7.5, ncol=1, loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n  Comparison plot saved to: {save_path}')
    plt.close()


def plot_best_per_noise(results, save_path='best_per_noise.png'):
    """
    Additional figure: for each noise level, show all three methods' best
    heights as points on a strip plot-style chart, to see separation clearly.
    """
    noise_labs  = list(NOISE_LEVELS.keys())
    noise_sigma = list(NOISE_LEVELS.values())
    methods     = ['BO', 'DE', 'CMA-ES']

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    fig.suptitle('Best Height per Noise Round — Individual Method Comparison',
                 fontsize=11, fontweight='bold')

    for ni, (nl, sigma) in enumerate(zip(noise_labs, noise_sigma)):
        ax = axes[ni]
        heights = {m: results[m][nl]['best_height'] * 100 for m in methods}
        thetas  = {m: results[m][nl]['best_theta'] for m in methods}

        for xi, method in enumerate(methods):
            ax.scatter([xi], [heights[method]],
                       color=COLORS[method], s=120, zorder=3,
                       edgecolors='white', linewidths=1)
            ax.text(xi, heights[method] + 0.5,
                    f'{heights[method]:.1f} cm',
                    ha='center', va='bottom', fontsize=8.5,
                    color=COLORS[method], fontweight='bold')
            # Show best angles below
            theta_str = '\n'.join([
                f'θ₁={thetas[method][0]:.2f}',
                f'θ₂={thetas[method][1]:.2f}',
                f'θ₃={thetas[method][2]:.2f}',
            ])
            ax.text(xi, ax.get_ylim()[0] if ni == 0 else 0,
                    theta_str, ha='center', va='top',
                    fontsize=6.5, color='#444', alpha=0.8)

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(f'{nl}\n(σ = {sigma:.2f} rad)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        if ni == 0:
            ax.set_ylabel('Best Jump Height (cm)', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Per-noise plot saved to: {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(results, path='optimization_results.csv'):
    rows = []
    for method, noise_dict in results.items():
        for noise_label, res in noise_dict.items():
            rows.append({
                'method':       method,
                'noise_label':  noise_label,
                'noise_std':    NOISE_LEVELS[noise_label],
                'best_height_m':  res['best_height'],
                'best_height_cm': res['best_height'] * 100,
                'theta1_rad':  res['best_theta'][0],
                'theta2_rad':  res['best_theta'][1],
                'theta3_rad':  res['best_theta'][2],
                'n_evals':     len(res['convergence']),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f'  Results CSV saved to: {path}')
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('=' * 65)
    print('  Multi-Method Jump Optimization')
    print('=' * 65)
    print(f'  Parameters optimized : theta1, theta2, theta3 (3 angles)')
    print(f'  Fixed tau_max        : ankle={TAU_MAX_ANKLE} N·m, '
          f'knee={TAU_MAX_KNEE} N·m, hip={TAU_MAX_HIP} N·m')
    print(f'  Noise rounds         : clean / low (σ=0.02) / high (σ=0.05)')
    print(f'  Methods              : Bayesian Opt, Differential Evolution, CMA-ES')
    print('=' * 65)

    t0 = time.time()

    results = run_all(
        n_initial_bo = 12,
        n_iter_bo    = 40,
        maxiter_de   = 40,
        popsize_de   = 6,
        maxiter_cma  = 40,
        sigma_cma    = 0.3,
        verbose      = True,
    )

    elapsed = time.time() - t0
    print(f'\nTotal wall time: {elapsed:.1f} s ({elapsed/60:.1f} min)')

    # ── Print table ──────────────────────────────────────────────────────────
    print_results_table(results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = save_results_csv(results, 'optimization_results.csv')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_results(results, 'optimization_comparison.png')
    plot_best_per_noise(results, 'best_per_noise.png')

    # ── Run final simulation with best overall result ─────────────────────────
    print('\n--- Best overall result across all methods & noise rounds ---')
    best_overall_h = 0.0
    best_overall_method = ''
    best_overall_noise  = ''
    best_overall_theta  = None

    for method, noise_dict in results.items():
        for noise_label, res in noise_dict.items():
            if res['best_height'] > best_overall_h:
                best_overall_h      = res['best_height']
                best_overall_method = method
                best_overall_noise  = noise_label
                best_overall_theta  = res['best_theta']

    print(f'  Method: {best_overall_method}  |  Noise: {best_overall_noise}')
    print(f'  Best height: {best_overall_h*100:.2f} cm')
    print(f'  Best angles: θ1={best_overall_theta[0]:.4f}, '
          f'θ2={best_overall_theta[1]:.4f}, θ3={best_overall_theta[2]:.4f}')

    h_final, t1, X1, t2, X2 = simulate_jump(
        best_overall_theta,
        return_trajectories=True,
        verbose=True
    )
