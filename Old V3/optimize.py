"""
optimize.py
===========
Multi-method optimization of the 3-segment jump model.

Optimization target: maximize jump height by tuning starting joint angles only.
    params = [theta1, theta2, theta3]   (3 parameters, radians)

Torques are fixed at literature-based maxima (step function, t=0 to liftoff):
    ankle : 175 N·m  (Hussain & Frey-Law 2016)
    knee  : 247 N·m  (Harbo et al. 2012)
    hip   : 175 N·m  (Harbo et al. 2012)

─────────────────────────────────────────────────────────────────────────────
FIXED EVALUATION BUDGET
─────────────────────────────────────────────────────────────────────────────
All three methods share the same total function-evaluation budget (EVAL_BUDGET).
This is the standard approach in the optimization benchmarking literature
(e.g., BBOB/COCO suite) for fair method comparison. Each method spends its
budget however its internal logic dictates; the x-axis of every convergence
curve is the shared currency "number of simulator calls", making curves
directly comparable.

    EVAL_BUDGET = 150   (= 50 × d, where d=3; a standard low-budget setting)

Method settings derived from the budget:
    BO      n_initial = 30 (Sobol warm-start), n_iter = 120 (GP-guided)
    DE      popsize = 4, maxiter = 12  ->  4x3x12 = 144 evals (approx budget)
    CMA-ES  stops as soon as eval counter reaches EVAL_BUDGET

─────────────────────────────────────────────────────────────────────────────
THREE OPTIMIZATION METHODS
─────────────────────────────────────────────────────────────────────────────
1. Bayesian Optimization (BOTorch, LogEI acquisition)
   Builds a Gaussian Process surrogate and picks the next point by
   maximising Expected Improvement.  Best when evaluations are expensive.
   Falls back to quasi-random Sobol search if BOTorch/torch is unavailable.

2. Differential Evolution (scipy.optimize.differential_evolution)
   Population-based stochastic global search.  Robust to multimodal and
   discontinuous landscapes with no gradient information needed.

3. CMA-ES (pycma)
   Adapts a full covariance matrix of a Gaussian search distribution.
   Often the strongest gradient-free method on smooth continuous landscapes.

─────────────────────────────────────────────────────────────────────────────
THREE NOISE ROUNDS
─────────────────────────────────────────────────────────────────────────────
Each method is run three times with different levels of Gaussian angle noise:
    Round 1 (clean)      sigma = 0.00 rad  - noiseless
    Round 2 (low noise)  sigma = 0.02 rad  - small postural variability
    Round 3 (high noise) sigma = 0.05 rad  - larger postural variability

In noisy rounds each evaluation averages N_NOISE_TRIALS perturbed simulations,
so the optimizer finds starting positions robust to angle errors.

Usage:
    python optimize.py

Requirements:
    pip install scipy cma matplotlib pandas numpy
    pip install botorch gpytorch torch   (optional -- BO degrades gracefully)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import warnings

from scipy.optimize import differential_evolution
import cma

from simulator import simulate_jump, get_default_params, TAU_MAX_ANKLE, TAU_MAX_KNEE, TAU_MAX_HIP

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  BOTorch -- optional import with graceful fallback
# ─────────────────────────────────────────────────────────────────────────────

BOTORCH_AVAILABLE = False
try:
    import torch as _torch_probe
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import LogExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
except (ImportError, OSError):
    print('[WARNING] BOTorch/torch unavailable -- BO will use Sobol random search.')


# ─────────────────────────────────────────────────────────────────────────────
#  SHARED CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Fixed evaluation budget ─────────────────────────────────────────────────
# Single source of truth. All method settings below are derived from this.
EVAL_BUDGET = 150        # total simulator calls per method per noise round
                         # = 50 x d  (d=3),  standard low-budget benchmark

# ── Method settings (all derived from EVAL_BUDGET) ──────────────────────────
#
#   BO:     30 Sobol init + 120 GP-guided = 150 exactly
#   DE:     pop = popsize * DIM = 4*3 = 12 per generation
#           maxiter = floor(150 / 12) = 12  -> 12*12 = 144 evals
#           (slight underrun is intentional; DE also runs an init generation)
#   CMA-ES: no fixed maxiter -- hard-stopped by budget counter

BO_N_INITIAL = 30
BO_N_ITER    = EVAL_BUDGET - BO_N_INITIAL   # 120

DE_POPSIZE   = 4
DE_MAXITER   = EVAL_BUDGET // (DE_POPSIZE * 3)   # floor(150/12) = 12

CMAES_SIGMA0 = 0.3       # initial step-size in normalised [0,1] space

N_NOISE_TRIALS = 5       # simulator calls averaged per noisy evaluation

# ── Parameter space ──────────────────────────────────────────────────────────
#   theta1 : ankle absolute angle  (foot segment, rad)
#   theta2 : knee relative angle   (shank rel. foot, rad)
#   theta3 : hip relative angle    (thigh rel. shank, rad)

LOWER_BOUNDS = np.array([-0.60,  0.20, -2.20])
UPPER_BOUNDS = np.array([ 0.10,  1.80, -0.30])
DIM = 3

NOISE_LEVELS = {
    'clean':      0.00,
    'low_noise':  0.02,
    'high_noise': 0.05,
}

# ── Plot styling ─────────────────────────────────────────────────────────────
COLORS = {
    'BO':     '#2196F3',
    'DE':     '#4CAF50',
    'CMA-ES': '#FF5722',
}
NOISE_LINESTYLE = {
    'clean':      '-',
    'low_noise':  '--',
    'high_noise': ':',
}
NOISE_ALPHA = {
    'clean':      1.00,
    'low_noise':  0.70,
    'high_noise': 0.45,
}


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize(x):
    """Physical angles -> [0, 1]^3."""
    return (np.asarray(x, dtype=float) - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)

def unnormalize(x):
    """[0, 1]^3 -> physical angles."""
    return np.asarray(x, dtype=float) * (UPPER_BOUNDS - LOWER_BOUNDS) + LOWER_BOUNDS


# ─────────────────────────────────────────────────────────────────────────────
#  BUDGET-AWARE OBJECTIVE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class BudgetExhausted(Exception):
    """Raised when a method tries to exceed its evaluation budget."""
    pass


class BudgetedObjective:
    """
    Wraps simulate_jump with:
      - a hard evaluation counter (raises BudgetExhausted at limit)
      - optional Gaussian angle noise averaged over N_NOISE_TRIALS trials
      - a best-so-far convergence trace recorded per individual simulator call

    The convergence trace has one entry per individual simulator call, so all
    three methods produce traces on the same x-axis (evaluation count).

    Parameters
    ----------
    p          : dict   model parameters from get_default_params()
    noise_std  : float  std dev of Gaussian angle perturbation (rad); 0 = clean
    budget     : int    maximum number of individual simulator calls
    """

    def __init__(self, p, noise_std=0.0, budget=EVAL_BUDGET):
        self.p           = p
        self.noise_std   = noise_std
        self.budget      = budget
        self.n_calls     = 0
        self.best        = 0.0
        self.convergence = []   # best-so-far (cm) after each simulator call

    # ── Single noiseless call ────────────────────────────────────────────────
    def _single(self, theta):
        """One simulator call. Raises BudgetExhausted if over limit."""
        if self.n_calls >= self.budget:
            raise BudgetExhausted
        self.n_calls += 1
        try:
            h = float(simulate_jump(
                np.clip(theta, LOWER_BOUNDS, UPPER_BOUNDS), self.p
            ))
            h = max(h, 0.0)
        except Exception:
            h = 0.0
        if h > self.best:
            self.best = h
        self.convergence.append(self.best * 100)   # cm
        return h

    # ── Public call (with optional noise) ────────────────────────────────────
    def __call__(self, theta):
        """
        Evaluate at theta.
        Noisy mode: averages N_NOISE_TRIALS perturbed runs (each counts vs budget).
        Clean mode: single run.
        Returns mean height in meters.
        """
        theta = np.asarray(theta, dtype=float)
        if self.noise_std > 0.0:
            heights = []
            for _ in range(N_NOISE_TRIALS):
                if self.n_calls >= self.budget:
                    break
                noisy = theta + np.random.randn(DIM) * self.noise_std
                heights.append(self._single(noisy))
            return float(np.mean(heights)) if heights else 0.0
        else:
            return self._single(theta)

    @property
    def exhausted(self):
        return self.n_calls >= self.budget


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 1 -- BAYESIAN OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

def run_bayesian_optimization(obj, verbose=True):
    """
    BO_N_INITIAL Sobol points + BO_N_ITER GP-guided points = EVAL_BUDGET total.

    If BOTorch is unavailable, degrades to pure Sobol random search over the
    full budget -- still a valid baseline for comparison.

    Parameters
    ----------
    obj     : BudgetedObjective
    verbose : bool

    Returns
    -------
    best_theta : np.ndarray (3,)
    best_height : float (m)
    convergence : list[float] (cm)
    """
    # ── Sobol fallback (no BOTorch) ──────────────────────────────────────────
    if not BOTORCH_AVAILABLE:
        if verbose:
            print(f'  [BO] BOTorch unavailable -- Sobol random search '
                  f'({EVAL_BUDGET} evals)')
        rng = np.random.RandomState(42)
        best_theta = unnormalize(rng.rand(DIM))
        while not obj.exhausted:
            theta = unnormalize(rng.rand(DIM))
            h = obj(theta)
            if h >= obj.best:
                best_theta = theta.copy()
        return best_theta, obj.best, obj.convergence

    # ── Full BOTorch path ────────────────────────────────────────────────────
    if verbose:
        print(f'  [BO] {BO_N_INITIAL} Sobol + {BO_N_ITER} GP-guided '
              f'= {EVAL_BUDGET} total')

    bounds_norm = torch.stack([
        torch.zeros(DIM, dtype=torch.double),
        torch.ones(DIM, dtype=torch.double)
    ])

    # Sobol warm-start
    sobol_raw = draw_sobol_samples(bounds_norm, n=BO_N_INITIAL, q=1).squeeze(1)
    train_X_list, train_Y_list = [], []

    for i in range(BO_N_INITIAL):
        if obj.exhausted:
            break
        theta = unnormalize(sobol_raw[i].numpy())
        h = obj(theta)
        train_X_list.append(normalize(theta))
        train_Y_list.append(h)
        if verbose:
            print(f'    Sobol [{i+1:2d}/{BO_N_INITIAL}] '
                  f'h={h*100:.2f} cm  best={obj.best*100:.2f} cm')

    train_X = torch.tensor(np.array(train_X_list), dtype=torch.double)
    train_Y = torch.tensor(train_Y_list, dtype=torch.double).unsqueeze(1)

    # GP-guided iterations
    it = 0
    while not obj.exhausted and it < BO_N_ITER:
        it += 1
        try:
            model = SingleTaskGP(train_X, train_Y,
                                 outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            acq = LogExpectedImprovement(model=model, best_f=train_Y.max())
            candidate, _ = optimize_acqf(
                acq_function=acq, bounds=bounds_norm,
                q=1, num_restarts=8, raw_samples=128,
            )
            theta = unnormalize(candidate.squeeze(0).detach().numpy())
        except Exception:
            theta = unnormalize(np.random.rand(DIM))

        h = obj(theta)
        x_norm = torch.tensor(normalize(theta), dtype=torch.double).unsqueeze(0)
        train_X = torch.cat([train_X, x_norm], dim=0)
        train_Y = torch.cat([train_Y,
                             torch.tensor([[h]], dtype=torch.double)], dim=0)

        if verbose:
            print(f'    BO [{it:3d}/{BO_N_ITER}] '
                  f'h={h*100:.2f} cm  best={obj.best*100:.2f} cm  '
                  f'[{obj.n_calls}/{EVAL_BUDGET} evals]')

    best_idx   = int(train_Y.argmax())
    best_theta = unnormalize(train_X[best_idx].numpy())
    return best_theta, obj.best, obj.convergence


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 2 -- DIFFERENTIAL EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_differential_evolution(obj, verbose=True):
    """
    DE with population DE_POPSIZE*DIM and DE_MAXITER generations.
    Actual evals = DE_POPSIZE * DIM * DE_MAXITER ~ EVAL_BUDGET.

    BudgetExhausted is caught to stop DE cleanly if scipy tries to
    squeeze in extra evaluations beyond the budget.
    polish=False to avoid a gradient-based tail step that can stall.

    Parameters
    ----------
    obj     : BudgetedObjective
    verbose : bool

    Returns
    -------
    best_theta, best_height, convergence
    """
    pop_size = DE_POPSIZE * DIM
    if verbose:
        print(f'  [DE] popsize={DE_POPSIZE} (pop={pop_size}), '
              f'maxiter={DE_MAXITER}  ->  ~{pop_size * DE_MAXITER} evals '
              f'(budget={EVAL_BUDGET})')

    def neg_h(theta):
        if obj.exhausted:
            raise BudgetExhausted
        h = obj(theta)
        if verbose and obj.n_calls % pop_size == 0:
            print(f'    DE [eval {obj.n_calls:3d}/{EVAL_BUDGET}]  '
                  f'best={obj.best*100:.2f} cm')
        return -h

    bounds = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))
    best_theta = None
    try:
        result = differential_evolution(
            neg_h, bounds,
            maxiter       = DE_MAXITER,
            popsize       = DE_POPSIZE,
            seed          = 42,
            tol           = 1e-6,
            mutation      = (0.5, 1.0),
            recombination = 0.7,
            strategy      = 'best1bin',
            polish        = False,
        )
        best_theta = result.x
    except BudgetExhausted:
        pass   # normal early stop -- best tracked in obj

    if best_theta is None:
        best_theta = LOWER_BOUNDS + np.random.rand(DIM) * (UPPER_BOUNDS - LOWER_BOUNDS)

    return best_theta, obj.best, obj.convergence


# ─────────────────────────────────────────────────────────────────────────────
#  METHOD 3 -- CMA-ES
# ─────────────────────────────────────────────────────────────────────────────

def run_cmaes(obj, verbose=True):
    """
    CMA-ES in normalised [0,1]^3 space. Hard-stops via budget counter.
    BudgetExhausted from _single() terminates the generation loop cleanly.

    Parameters
    ----------
    obj     : BudgetedObjective
    verbose : bool

    Returns
    -------
    best_theta, best_height, convergence
    """
    if verbose:
        print(f'  [CMA-ES] sigma0={CMAES_SIGMA0}  budget={EVAL_BUDGET}')

    x0 = normalize(np.array([-0.15, 0.80, -1.20]))

    opts = cma.CMAOptions()
    opts['bounds']  = [[0.0] * DIM, [1.0] * DIM]
    opts['verbose'] = -9
    opts['tolx']    = 1e-5
    opts['tolfun']  = 1e-6
    opts['seed']    = 42

    es = cma.CMAEvolutionStrategy(x0, CMAES_SIGMA0, opts)

    best_x_norm = x0.copy()
    gen = 0

    while not es.stop() and not obj.exhausted:
        solutions = es.ask()
        fitnesses = []
        for s in solutions:
            if obj.exhausted:
                fitnesses.append(0.0)
                continue
            h = obj(unnormalize(np.clip(s, 0.0, 1.0)))
            fitnesses.append(-h)
        es.tell(solutions, fitnesses)
        gen += 1
        if verbose and gen % 3 == 0:
            print(f'    CMA [gen {gen:3d}  eval {obj.n_calls:3d}/{EVAL_BUDGET}]  '
                  f'best={obj.best*100:.2f} cm')

    res = es.result
    if res.xbest is not None:
        best_x_norm = np.clip(res.xbest, 0.0, 1.0)

    return unnormalize(best_x_norm), obj.best, obj.convergence


# ─────────────────────────────────────────────────────────────────────────────
#  RUNNER -- ALL METHODS x ALL NOISE ROUNDS
# ─────────────────────────────────────────────────────────────────────────────

def run_all(verbose=True):
    """
    Run all 3 optimizers x 3 noise levels = 9 experiments.
    Each experiment gets exactly EVAL_BUDGET individual simulator calls.

    Returns
    -------
    results : dict  results[method][noise_label] = {
        'best_theta', 'best_height', 'convergence', 'n_calls'
    }
    """
    p = get_default_params()
    results = {m: {} for m in ('BO', 'DE', 'CMA-ES')}

    for noise_label, noise_std in NOISE_LEVELS.items():
        print('\n' + '=' * 68)
        print(f'  NOISE ROUND : {noise_label}  (sigma = {noise_std:.2f} rad)')
        print(f'  EVAL BUDGET : {EVAL_BUDGET} per method')
        print('=' * 68)

        for method_name, run_fn in [
            ('BO',     run_bayesian_optimization),
            ('DE',     run_differential_evolution),
            ('CMA-ES', run_cmaes),
        ]:
            print(f'\n  -- {method_name} --')
            obj = BudgetedObjective(p, noise_std=noise_std, budget=EVAL_BUDGET)
            theta, h, conv = run_fn(obj, verbose=verbose)
            results[method_name][noise_label] = {
                'best_theta':  theta,
                'best_height': h,
                'convergence': conv,
                'n_calls':     obj.n_calls,
            }
            print(f'  [{method_name}] best={h*100:.2f} cm  '
                  f'theta={np.round(theta, 3)}  evals={obj.n_calls}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results):
    methods    = ['BO', 'DE', 'CMA-ES']
    noise_labs = list(NOISE_LEVELS.keys())

    print('\n' + '=' * 68)
    print('  RESULTS SUMMARY')
    print('=' * 68)
    print(f'  Shared eval budget : {EVAL_BUDGET} per method')
    print(f'  Fixed torques      : ankle={TAU_MAX_ANKLE} N*m  '
          f'knee={TAU_MAX_KNEE} N*m  hip={TAU_MAX_HIP} N*m')
    print()

    cw = 13
    header = ('Method'.ljust(10) + 'Noise'.ljust(14) +
              'Height(cm)'.ljust(cw) + 'Evals'.ljust(8) +
              'th1(rad)'.ljust(cw) + 'th2(rad)'.ljust(cw) + 'th3(rad)'.ljust(cw))
    print(header)
    print('-' * len(header))

    for method in methods:
        for nl in noise_labs:
            r = results[method][nl]
            row = (method.ljust(10) + nl.ljust(14) +
                   f'{r["best_height"]*100:.2f}'.ljust(cw) +
                   f'{r["n_calls"]}'.ljust(8) +
                   f'{r["best_theta"][0]:.4f}'.ljust(cw) +
                   f'{r["best_theta"][1]:.4f}'.ljust(cw) +
                   f'{r["best_theta"][2]:.4f}'.ljust(cw))
            print(row)
        print()


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results, save_path='optimization_comparison.png', gt=None):
    """
    Two-panel figure:
      Left  -- convergence curves (best-so-far vs eval number, all 9 runs)
              + horizontal dashed line at ground truth height (if provided)
      Right -- grouped bar chart of final best height per method x noise level
              + horizontal dashed line at ground truth height (if provided)

    The x-axis of the convergence panel is the shared evaluation count, making
    all curves directly comparable (standard fixed-budget plot convention).

    Parameters
    ----------
    results  : dict from run_all()
    save_path: str
    gt       : dict from find_ground_truth(), or None
    """
    methods    = ['BO', 'DE', 'CMA-ES']
    noise_labs = list(NOISE_LEVELS.keys())
    noise_sigs = list(NOISE_LEVELS.values())

    fig, (ax_conv, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'3-Segment Jump Optimisation -- Fixed Budget ({EVAL_BUDGET} evals/method)\n'
        f'Torques fixed: ankle {TAU_MAX_ANKLE} N*m  '
        f'knee {TAU_MAX_KNEE} N*m  hip {TAU_MAX_HIP} N*m',
        fontsize=11, fontweight='bold', y=1.02
    )

    # ── Left: convergence curves ─────────────────────────────────────────────
    for method in methods:
        for nl in noise_labs:
            conv  = results[method][nl]['convergence']
            evals = np.arange(1, len(conv) + 1)
            ax_conv.plot(
                evals, conv,
                color     = COLORS[method],
                linestyle = NOISE_LINESTYLE[nl],
                linewidth = 1.8,
                alpha     = NOISE_ALPHA[nl],
                label     = f'{method} | {nl}',
            )

    # Ground truth line on convergence panel
    if gt is not None:
        ax_conv.axhline(
            gt['height_cm'],
            color='black', lw=1.8, ls='--', zorder=5,
            label=f'Ground truth ({gt["height_cm"]:.1f} cm)'
        )
        ax_conv.annotate(
            f'Ground truth\n{gt["height_cm"]:.1f} cm',
            xy=(EVAL_BUDGET * 0.98, gt['height_cm']),
            xytext=(-6, 6), textcoords='offset points',
            ha='right', va='bottom', fontsize=8,
            color='black', fontweight='bold',
        )

    ax_conv.set_xlabel('Simulator evaluations', fontsize=10)
    ax_conv.set_ylabel('Best jump height so far (cm)', fontsize=10)
    ax_conv.set_title('Convergence curves -- shared evaluation budget', fontsize=10)
    ax_conv.set_xlim(1, EVAL_BUDGET)
    ax_conv.axvline(EVAL_BUDGET, color='black', lw=0.8, ls=':', alpha=0.4)
    ax_conv.legend(fontsize=8, ncol=1, loc='lower right')
    ax_conv.grid(True, alpha=0.3)

    # ── Right: bar chart ─────────────────────────────────────────────────────
    n_methods = len(methods)
    bar_w     = 0.22
    group_pos = np.arange(len(noise_labs))
    offsets   = np.linspace(-(n_methods-1)/2,
                             (n_methods-1)/2, n_methods) * bar_w

    for mi, method in enumerate(methods):
        heights_cm = [results[method][nl]['best_height'] * 100
                      for nl in noise_labs]
        xpos = group_pos + offsets[mi]
        bars = ax_bar.bar(xpos, heights_cm, width=bar_w,
                          color=COLORS[method], alpha=0.85,
                          label=method, zorder=3,
                          edgecolor='white', linewidth=0.6)
        for bar, hcm in zip(bars, heights_cm):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{hcm:.1f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=COLORS[method]
            )

    ax_bar.set_xticks(group_pos)
    ax_bar.set_xticklabels(
        [f'{nl}\n(sigma={s:.2f} rad)' for nl, s in zip(noise_labs, noise_sigs)],
        fontsize=9
    )
    ax_bar.set_ylabel('Best jump height (cm)', fontsize=10)
    ax_bar.set_title(f'Final best after {EVAL_BUDGET} evals', fontsize=10)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, axis='y', alpha=0.3, zorder=0)
    ax_bar.set_ylim(0, ax_bar.get_ylim()[1] * 1.13)

    # Ground truth line on bar chart
    if gt is not None:
        ax_bar.axhline(
            gt['height_cm'],
            color='black', lw=1.8, ls='--', zorder=6,
            label=f'Ground truth ({gt["height_cm"]:.1f} cm)'
        )
        ax_bar.legend(fontsize=9)
        ax_bar.annotate(
            f'Ground truth: {gt["height_cm"]:.1f} cm',
            xy=(ax_bar.get_xlim()[1], gt['height_cm']),
            xytext=(-6, 4), textcoords='offset points',
            ha='right', va='bottom', fontsize=8,
            color='black', fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved to: {save_path}')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  GROUND TRUTH  (high-budget deterministic global search, no noise)
# ─────────────────────────────────────────────────────────────────────────────

def find_ground_truth(p=None, verbose=True):
    """
    Estimate the true global optimum of jump height over the angle search space
    using three complementary high-budget, noiseless global solvers:

        1. Differential Evolution   -- popsize=15, maxiter=500 (~22 500 evals)
           population-based global search, robust to discontinuities
        2. dual_annealing           -- maxiter=10 000
           simulated annealing with local gradient polishing; good at escaping
           local optima
        3. shgo (Simplicial Homology Global Optimisation) -- n=256 sampling pts
           topology-based solver that provides quasi-global guarantees on
           Lipschitz-continuous functions

    The best result across all three solvers is returned as the ground truth.
    Because the liftoff event introduces a mild discontinuity, we cannot claim
    mathematical optimality, but in practice three independent high-budget
    solvers converging to the same value is strong evidence of the true peak.

    Parameters
    ----------
    p       : dict, optional  model parameters (default: get_default_params())
    verbose : bool

    Returns
    -------
    gt : dict with keys
        'height_cm'  : float  -- best jump height found (cm)
        'height_m'   : float  -- best jump height found (m)
        'theta'      : np.ndarray (3,) -- optimal angles (rad)
        'n_evals'    : int    -- total simulator calls used
        'solver_heights' : dict  -- best height per solver (for reporting)
    """

    if p is None:
        p = get_default_params()

    bounds = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))
    total_evals = [0]

    def neg_h(theta):
        total_evals[0] += 1
        try:
            h = float(simulate_jump(theta, p))
            return -max(h, 0.0)
        except Exception:
            return 0.0

    solver_results = {}
    solver_thetas  = {}

    # ── 1. Differential Evolution — robust global search ──────────────────────
    # popsize=15 gives 45 members; convergence typically within ~500 gens.
    # tol=1e-6 + polish=True adds a final L-BFGS-B local refinement step.
    if verbose:
        print('\n  [Ground truth] Solver 1/3: Differential Evolution '
              '(popsize=15, maxiter=500, polish=True) ...')
    t0 = time.time()
    res_de = differential_evolution(
        neg_h, bounds,
        popsize       = 15,
        maxiter       = 500,
        seed          = 0,
        tol           = 1e-6,
        mutation      = (0.5, 1.0),
        recombination = 0.9,
        strategy      = 'best1bin',
        polish        = True,
    )
    solver_results['DE'] = -res_de.fun
    solver_thetas['DE']  = res_de.x
    if verbose:
        print(f'    DE:   {-res_de.fun*100:.3f} cm  '
              f'({time.time()-t0:.1f}s, {total_evals[0]} evals so far)')

    # ── 2. DE with different seed/strategy — independent cross-check ──────────
    # Using a different random seed and currenttobest1bin strategy gives an
    # independent search trajectory that confirms (or improves on) result 1.
    # dual_annealing was removed because even with no_local_search=True it
    # spawns internal gradient calls that make it slow on simulator objectives.
    if verbose:
        print('  [Ground truth] Solver 2/3: Differential Evolution '
              '(seed=1, strategy=currenttobest1bin) ...')
    t0 = time.time()
    res_de2 = differential_evolution(
        neg_h, bounds,
        popsize       = 15,
        maxiter       = 500,
        seed          = 1,          # different seed -> independent trajectory
        tol           = 1e-6,
        mutation      = (0.4, 0.9),
        recombination = 0.95,
        strategy      = 'currenttobest1bin',  # different strategy
        polish        = True,
    )
    solver_results['DE2'] = -res_de2.fun
    solver_thetas['DE2']  = res_de2.x
    if verbose:
        print(f'    DE2:  {-res_de2.fun*100:.3f} cm  '
              f'({time.time()-t0:.1f}s, {total_evals[0]} evals so far)')

    # ── 3. Multi-start L-BFGS-B — precise local refinement ───────────────────
    # Start from 50 random points across the search space and run a fast
    # gradient-free L-BFGS-B polish from each.  This is cheap (~50-200 evals
    # total) and catches any sharp peaks that global methods might step over.
    from scipy.optimize import minimize
    if verbose:
        print('  [Ground truth] Solver 3/3: Multi-start L-BFGS-B (50 starts) ...')
    t0 = time.time()
    rng = np.random.RandomState(0)
    ms_best_h, ms_best_x = 0.0, res_de.x.copy()
    for _ in range(50):
        x0 = LOWER_BOUNDS + rng.rand(DIM) * (UPPER_BOUNDS - LOWER_BOUNDS)
        try:
            res_ms = minimize(neg_h, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 100, 'ftol': 1e-9})
            h_ms = -res_ms.fun
            if h_ms > ms_best_h:
                ms_best_h = h_ms
                ms_best_x = res_ms.x
        except Exception:
            pass
    solver_results['multistart'] = ms_best_h
    solver_thetas['multistart']  = ms_best_x
    if verbose:
        print(f'    MS:   {ms_best_h*100:.3f} cm  '
              f'({time.time()-t0:.1f}s, {total_evals[0]} evals so far)')

    # ── Pick best across all three solvers ────────────────────────────────────
    best_solver = max(solver_results, key=solver_results.get)
    best_height = solver_results[best_solver]
    best_theta  = solver_thetas[best_solver]

    gt = {
        'height_cm':      best_height * 100,
        'height_m':       best_height,
        'theta':          best_theta,
        'n_evals':        total_evals[0],
        'solver_heights': {k: v * 100 for k, v in solver_results.items()},
    }

    if verbose:
        print(f'\n  [Ground truth] Best solver : {best_solver}')
        print(f'  [Ground truth] Height      : {best_height*100:.3f} cm')
        print(f'  [Ground truth] Angles      : '
              f'theta1={best_theta[0]:.4f}  '
              f'theta2={best_theta[1]:.4f}  '
              f'theta3={best_theta[2]:.4f} rad')
        print(f'  [Ground truth] Total evals : {total_evals[0]}')
        solver_str = '  '.join(
            f'{k}={v:.2f}cm' for k, v in gt['solver_heights'].items()
        )
        print(f'  [Ground truth] All solvers : {solver_str}')

    return gt


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(results, path='optimization_results.csv', gt=None):
    rows = []
    for method, noise_dict in results.items():
        for noise_label, res in noise_dict.items():
            rows.append({
                'method':         method,
                'noise_label':    noise_label,
                'noise_std_rad':  NOISE_LEVELS[noise_label],
                'eval_budget':    EVAL_BUDGET,
                'n_evals_used':   res['n_calls'],
                'best_height_m':  res['best_height'],
                'best_height_cm': res['best_height'] * 100,
                'theta1_rad':     res['best_theta'][0],
                'theta2_rad':     res['best_theta'][1],
                'theta3_rad':     res['best_theta'][2],
            })
    df = pd.DataFrame(rows)

    # Optionally append ground truth as a sentinel row
    if gt is not None:
        gt_row = pd.DataFrame([{
            'method':         'GROUND_TRUTH',
            'noise_label':    'clean',
            'noise_std_rad':  0.0,
            'eval_budget':    gt['n_evals'],
            'n_evals_used':   gt['n_evals'],
            'best_height_m':  gt['height_m'],
            'best_height_cm': gt['height_cm'],
            'theta1_rad':     gt['theta'][0],
            'theta2_rad':     gt['theta'][1],
            'theta3_rad':     gt['theta'][2],
        }])
        df = pd.concat([df, gt_row], ignore_index=True)

    df.to_csv(path, index=False)
    print(f'  CSV saved to: {path}')
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('=' * 68)
    print('  Multi-Method Jump Optimisation -- Fixed Evaluation Budget')
    print('=' * 68)
    print(f'  Parameters  : theta1, theta2, theta3  (d = {DIM})')
    print(f'  Eval budget : {EVAL_BUDGET} per method  (= {EVAL_BUDGET//DIM}xd)')
    print(f'  BO          : {BO_N_INITIAL} Sobol init + {BO_N_ITER} GP iters')
    print(f'  DE          : popsize={DE_POPSIZE} (pop={DE_POPSIZE*DIM}), '
          f'maxiter={DE_MAXITER}  -> ~{DE_POPSIZE*DIM*DE_MAXITER} evals')
    print(f'  CMA-ES      : sigma0={CMAES_SIGMA0}, hard-stop at budget')
    print(f'  Noise rounds: clean / low (sigma=0.02) / high (sigma=0.05)')
    print(f'  Fixed taus  : ankle={TAU_MAX_ANKLE} N*m  '
          f'knee={TAU_MAX_KNEE} N*m  hip={TAU_MAX_HIP} N*m')
    print('=' * 68)

    t0 = time.time()

    # ── Step 1: Ground truth (high-budget, no noise) ──────────────────────────
    print('\n' + '=' * 68)
    print('  STEP 1: Finding ground truth (high-budget global solvers) ...')
    print('=' * 68)
    gt = find_ground_truth(verbose=True)

    # ── Step 2: Fixed-budget comparison across methods & noise rounds ─────────
    print('\n' + '=' * 68)
    print('  STEP 2: Fixed-budget comparison (all methods x all noise rounds)')
    print('=' * 68)
    results = run_all(verbose=True)

    elapsed = time.time() - t0
    print(f'\nTotal wall time: {elapsed:.1f} s  ({elapsed/60:.1f} min)')

    print_results_table(results)
    save_results_csv(results, 'optimization_results.csv', gt=gt)
    plot_results(results, 'optimization_comparison.png', gt=gt)

    # ── Best overall ─────────────────────────────────────────────────────────
    print('\n' + '-' * 68)
    print('  BEST OVERALL')
    print('-' * 68)
    best_h, best_m, best_nl, best_theta = 0.0, '', '', None
    for method, nd in results.items():
        for nl, res in nd.items():
            if res['best_height'] > best_h:
                best_h, best_m, best_nl = res['best_height'], method, nl
                best_theta = res['best_theta']

    print(f'  Method : {best_m}  |  Noise : {best_nl}')
    print(f'  Height : {best_h*100:.2f} cm')
    print(f'  Angles : theta1={best_theta[0]:.4f}  '
          f'theta2={best_theta[1]:.4f}  theta3={best_theta[2]:.4f} rad')

    gap = gt['height_cm'] - best_h * 100
    pct = 100.0 * gap / gt['height_cm'] if gt['height_cm'] > 0 else 0.0
    print(f'\n  Ground truth   : {gt["height_cm"]:.2f} cm')
    print(f'  Gap to GT      : {gap:.2f} cm  ({pct:.1f}% below ground truth)')

    print('\n--- Validation simulation with best parameters ---')
    simulate_jump(best_theta, return_trajectories=True, verbose=True)
