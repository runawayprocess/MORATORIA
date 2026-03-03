"""
Sobol global sensitivity analysis for the Moratoria Impact Model.

Performs variance-based decomposition (Sobol 2001) to identify which
parameters drive the most variance in model outputs. Uses Saltelli
sampling (SALib) across 8 key parameters, running the full A->B->C
pipeline for each sample.

Output: first-order indices (S1, direct effect) and total-order indices
(ST, including interactions) for three metrics:
  - Peak delay (weeks)
  - Peak capacity shortfall (%)
  - Cumulative compute deficit (%)

References:
- Sobol (2001): Global sensitivity indices for nonlinear models
- Saltelli et al. (2010): Variance-based sensitivity analysis
- SALib: Sensitivity Analysis Library in Python (Herman & Usher)
"""

import numpy as np
from dataclasses import dataclass

# Parameter modules to monkey-patch
import moratoria.module_a.displacement as disp_mod
import moratoria.module_b.capacity as cap_mod
import moratoria.module_c.compute as comp_mod
import moratoria.config as config_mod

from moratoria.data.scenarios import SCENARIOS
from moratoria.simulation.runner import run_scenario
from moratoria.module_c.compute import compare_to_baseline


# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "investment_elasticity",
    "reallocation_delay",
    "algo_doubling_months",
    "effective_fungibility",
    "congestion_sensitivity",
    "hardware_improvement_qtr",
    "agglomeration_elasticity",
    "logit_temperature",
]

PARAM_BOUNDS = [
    [0.10, 0.35],    # investment_elasticity
    [3.0, 8.0],      # reallocation_delay (treated as continuous, rounded in model)
    [8.0, 18.0],     # algo_doubling_months
    [0.40, 0.70],    # effective_fungibility
    [0.10, 0.60],    # congestion_sensitivity
    [0.06, 0.12],    # hardware_improvement_qtr
    [0.20, 0.70],    # agglomeration_elasticity
    [0.08, 0.25],    # logit_temperature
]

PARAM_LABELS = {
    "investment_elasticity": "Inv. Elasticity",
    "reallocation_delay": "Realloc. Delay",
    "algo_doubling_months": "Algo Doubling Time",
    "effective_fungibility": "Fungibility",
    "congestion_sensitivity": "Queue Congestion",
    "hardware_improvement_qtr": "HW Improvement/Qtr",
    "agglomeration_elasticity": "Agglom. Elasticity",
    "logit_temperature": "Logit Temperature",
}


@dataclass
class SobolResults:
    """Results from Sobol analysis."""
    param_names: list[str]
    s1: dict[str, np.ndarray]     # first-order indices per metric
    st: dict[str, np.ndarray]     # total-order indices per metric
    s1_conf: dict[str, np.ndarray]
    st_conf: dict[str, np.ndarray]
    n_samples: int
    metric_names: list[str]


# ---------------------------------------------------------------------------
# Model runner with parameter overrides
# ---------------------------------------------------------------------------

def _patch_and_run(params: dict) -> dict:
    """
    Run one model evaluation with parameter overrides.

    Monkey-patches module-level variables, runs the simulation,
    restores originals. This is necessary because the model reads
    parameters at function call time from module-level variables.
    """
    # Save originals
    originals = {
        "disp_fungibility": disp_mod.EFFECTIVE_FUNGIBILITY,
        "disp_temperature": disp_mod.LOGIT_TEMPERATURE,
        "disp_agglom_elast": disp_mod.AGGLOMERATION_ELASTICITY,
        "cap_congestion": cap_mod.CONGESTION_SENSITIVITY,
        "comp_hw_improve": comp_mod.HARDWARE_IMPROVEMENT_QTR,
        "comp_algo_doubling": comp_mod.ALGO_DOUBLING_TIME_MONTHS,
    }

    try:
        # Patch
        disp_mod.EFFECTIVE_FUNGIBILITY = params["effective_fungibility"]
        disp_mod.LOGIT_TEMPERATURE = params["logit_temperature"]
        disp_mod.AGGLOMERATION_ELASTICITY = params["agglomeration_elasticity"]
        cap_mod.CONGESTION_SENSITIVITY = params["congestion_sensitivity"]
        comp_mod.HARDWARE_IMPROVEMENT_QTR = params["hardware_improvement_qtr"]
        comp_mod.ALGO_DOUBLING_TIME_MONTHS = params["algo_doubling_months"]

        # Run baseline
        bl = run_scenario(
            SCENARIOS["baseline"],
            reallocation_delay=int(round(params["reallocation_delay"])),
            investment_elasticity=params["investment_elasticity"],
        )

        # Run scenario
        sc = run_scenario(
            SCENARIOS["all_dem_trifectas"],
            reallocation_delay=int(round(params["reallocation_delay"])),
            investment_elasticity=params["investment_elasticity"],
        )

        compare_to_baseline(bl.ai_timeline, sc.ai_timeline)

        peak_delay_wk = sc.ai_timeline.peak_delay_qtrs * 13 if sc.ai_timeline.peak_delay_qtrs else 0
        peak_shortfall = float(sc.ai_timeline.compute_shortfall_pct.max()) if sc.ai_timeline.compute_shortfall_pct is not None else 0
        cumul_deficit = sc.ai_timeline.cumulative_flop_deficit_pct if sc.ai_timeline.cumulative_flop_deficit_pct is not None else 0

        return {
            "peak_delay_wk": peak_delay_wk,
            "peak_shortfall_pct": peak_shortfall,
            "cumul_deficit_pct": cumul_deficit,
        }

    finally:
        # Restore
        disp_mod.EFFECTIVE_FUNGIBILITY = originals["disp_fungibility"]
        disp_mod.LOGIT_TEMPERATURE = originals["disp_temperature"]
        disp_mod.AGGLOMERATION_ELASTICITY = originals["disp_agglom_elast"]
        cap_mod.CONGESTION_SENSITIVITY = originals["cap_congestion"]
        comp_mod.HARDWARE_IMPROVEMENT_QTR = originals["comp_hw_improve"]
        comp_mod.ALGO_DOUBLING_TIME_MONTHS = originals["comp_algo_doubling"]


# ---------------------------------------------------------------------------
# Sobol analysis
# ---------------------------------------------------------------------------

def run_sobol(n_samples: int = 128, verbose: bool = True) -> SobolResults:
    """
    Run full Sobol global sensitivity analysis.

    Args:
        n_samples: Base sample size. Total runs = n_samples * (2*D + 2)
                   where D=8 parameters. Default 128 -> 2304 runs.
        verbose: Print progress updates.

    Returns:
        SobolResults with first-order and total-order indices.
    """
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    problem = {
        "num_vars": len(PARAM_NAMES),
        "names": PARAM_NAMES,
        "bounds": PARAM_BOUNDS,
    }

    if verbose:
        total_runs = n_samples * (2 * len(PARAM_NAMES) + 2)
        print(f"  Generating Saltelli samples: N={n_samples}, D={len(PARAM_NAMES)}, total runs={total_runs}")

    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
    total_runs = len(param_values)

    if verbose:
        print(f"  Running {total_runs} model evaluations...")

    # Collect outputs
    metric_names = ["peak_delay_wk", "peak_shortfall_pct", "cumul_deficit_pct"]
    outputs = {m: np.zeros(total_runs) for m in metric_names}

    for i, sample in enumerate(param_values):
        if verbose and (i + 1) % 100 == 0:
            print(f"    {i + 1}/{total_runs} complete...")

        params = dict(zip(PARAM_NAMES, sample))
        result = _patch_and_run(params)

        for m in metric_names:
            outputs[m][i] = result[m]

    if verbose:
        print(f"  Analyzing Sobol indices...")

    # Compute Sobol indices for each metric
    s1_all = {}
    st_all = {}
    s1_conf_all = {}
    st_conf_all = {}

    for m in metric_names:
        si = sobol.analyze(problem, outputs[m], calc_second_order=False)
        s1_all[m] = si["S1"]
        st_all[m] = si["ST"]
        s1_conf_all[m] = si["S1_conf"]
        st_conf_all[m] = si["ST_conf"]

    return SobolResults(
        param_names=PARAM_NAMES,
        s1=s1_all,
        st=st_all,
        s1_conf=s1_conf_all,
        st_conf=st_conf_all,
        n_samples=n_samples,
        metric_names=metric_names,
    )


def print_sobol_results(results: SobolResults):
    """Print formatted Sobol sensitivity indices."""
    print("=" * 72)
    print("  SOBOL GLOBAL SENSITIVITY ANALYSIS")
    print("  (All Democratic Trifectas scenario)")
    print("=" * 72)
    print(f"\n  Base samples: {results.n_samples}, Total model runs: {results.n_samples * (2 * len(results.param_names) + 2)}")
    print(f"  Parameters: {len(results.param_names)}")

    metric_labels = {
        "peak_delay_wk": "Peak Delay (weeks)",
        "peak_shortfall_pct": "Peak Shortfall (%)",
        "cumul_deficit_pct": "Cumulative Deficit (%)",
    }

    for metric in results.metric_names:
        label = metric_labels.get(metric, metric)
        print(f"\n  --- {label} ---")
        print(f"  {'Parameter':<22} | {'S1':>6} {'(conf)':>8} | {'ST':>6} {'(conf)':>8} |")
        print(f"  {'-' * 58}")

        # Sort by ST descending
        order = np.argsort(-results.st[metric])

        for idx in order:
            name = PARAM_LABELS.get(results.param_names[idx], results.param_names[idx])
            s1 = results.s1[metric][idx]
            s1_c = results.s1_conf[metric][idx]
            st = results.st[metric][idx]
            st_c = results.st_conf[metric][idx]
            print(f"  {name:<22} | {s1:>6.3f} ({s1_c:>6.3f}) | {st:>6.3f} ({st_c:>6.3f}) |")

    print()
    print("  S1 = first-order (direct effect); ST = total-order (including interactions)")
    print("  Parameters with ST > 0.1 are meaningfully influential.")
    print("  Large ST-S1 gap indicates parameter interacts strongly with others.")
    print()
