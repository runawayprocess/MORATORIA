"""
Sobol global sensitivity analysis for the Moratoria Impact Model.

Performs variance-based decomposition (Sobol 2001) to identify which
parameters drive the most variance in model outputs. Uses Saltelli
sampling (SALib) across 8 key parameters, running the full A->B->C
pipeline for each sample.

Output: first-order indices (S1, direct effect), total-order indices
(ST, including interactions), and optionally second-order indices (S2,
pairwise interactions) for three metrics:
  - Peak delay (weeks)
  - Peak capacity shortfall (%)
  - Cumulative compute deficit (%)

References:
- Sobol (2001): Global sensitivity indices for nonlinear models
- Saltelli et al. (2010): Variance-based sensitivity analysis
- SALib: Sensitivity Analysis Library in Python (Herman & Usher)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

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
    "fungibility_price_response",
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
    [0.00, 1.00],    # fungibility_price_response (0 = static, 1 = strong price response)
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
    "fungibility_price_response": "Fung. Price Resp.",
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
    # Second-order (optional)
    s2: Optional[dict[str, np.ndarray]] = None
    s2_conf: Optional[dict[str, np.ndarray]] = None
    # Raw outputs for percentile computation
    outputs: Optional[dict[str, np.ndarray]] = None


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
        "disp_fung_price": disp_mod.FUNGIBILITY_PRICE_RESPONSE,
        "cap_congestion": cap_mod.CONGESTION_SENSITIVITY,
        "comp_hw_improve": comp_mod.HARDWARE_IMPROVEMENT_QTR,
        "comp_algo_doubling": comp_mod.ALGO_DOUBLING_TIME_MONTHS,
    }

    try:
        # Patch
        disp_mod.EFFECTIVE_FUNGIBILITY = params["effective_fungibility"]
        disp_mod.LOGIT_TEMPERATURE = params["logit_temperature"]
        disp_mod.AGGLOMERATION_ELASTICITY = params["agglomeration_elasticity"]
        disp_mod.FUNGIBILITY_PRICE_RESPONSE = params["fungibility_price_response"]
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
        disp_mod.FUNGIBILITY_PRICE_RESPONSE = originals["disp_fung_price"]
        cap_mod.CONGESTION_SENSITIVITY = originals["cap_congestion"]
        comp_mod.HARDWARE_IMPROVEMENT_QTR = originals["comp_hw_improve"]
        comp_mod.ALGO_DOUBLING_TIME_MONTHS = originals["comp_algo_doubling"]


# ---------------------------------------------------------------------------
# Sobol analysis
# ---------------------------------------------------------------------------

def run_sobol(
    n_samples: int = 128,
    calc_second_order: bool = True,
    verbose: bool = True,
) -> SobolResults:
    """
    Run full Sobol global sensitivity analysis.

    Args:
        n_samples: Base sample size. Total runs depends on calc_second_order:
                   with second_order: n_samples * (2*D + 2)
                   without: n_samples * (D + 2)
        calc_second_order: Compute pairwise interaction indices (S2).
        verbose: Print progress updates.

    Returns:
        SobolResults with first-order, total-order, and optionally
        second-order indices plus raw outputs for percentile analysis.
    """
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    problem = {
        "num_vars": len(PARAM_NAMES),
        "names": PARAM_NAMES,
        "bounds": PARAM_BOUNDS,
    }

    if calc_second_order:
        total_formula = n_samples * (2 * len(PARAM_NAMES) + 2)
    else:
        total_formula = n_samples * (len(PARAM_NAMES) + 2)

    if verbose:
        print(f"  Generating Saltelli samples: N={n_samples}, D={len(PARAM_NAMES)}, "
              f"second_order={calc_second_order}, total runs={total_formula}")

    param_values = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
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
    s2_all = {} if calc_second_order else None
    s2_conf_all = {} if calc_second_order else None

    for m in metric_names:
        si = sobol.analyze(problem, outputs[m], calc_second_order=calc_second_order)
        s1_all[m] = si["S1"]
        st_all[m] = si["ST"]
        s1_conf_all[m] = si["S1_conf"]
        st_conf_all[m] = si["ST_conf"]
        if calc_second_order:
            s2_all[m] = si["S2"]
            s2_conf_all[m] = si["S2_conf"]

    return SobolResults(
        param_names=PARAM_NAMES,
        s1=s1_all,
        st=st_all,
        s1_conf=s1_conf_all,
        st_conf=st_conf_all,
        n_samples=n_samples,
        metric_names=metric_names,
        s2=s2_all,
        s2_conf=s2_conf_all,
        outputs=outputs,
    )


def print_sobol_results(results: SobolResults):
    """Print formatted Sobol sensitivity indices with percentile ranges."""
    print("=" * 72)
    print("  SOBOL GLOBAL SENSITIVITY ANALYSIS")
    print("  (All Democratic Trifectas scenario)")
    print("=" * 72)

    total_runs = results.n_samples * (2 * len(results.param_names) + 2)
    print(f"\n  Base samples: {results.n_samples}, Total model runs: {total_runs}")
    print(f"  Parameters: {len(results.param_names)}")

    metric_labels = {
        "peak_delay_wk": "Peak Delay (weeks)",
        "peak_shortfall_pct": "Peak Shortfall (%)",
        "cumul_deficit_pct": "Cumulative Deficit (%)",
    }

    # First/total order indices
    for metric in results.metric_names:
        label = metric_labels.get(metric, metric)
        print(f"\n  --- {label} ---")
        print(f"  {'Parameter':<22} | {'S1':>6} {'(conf)':>8} | {'ST':>6} {'(conf)':>8} |")
        print(f"  {'-' * 58}")

        order = np.argsort(-results.st[metric])
        for idx in order:
            name = PARAM_LABELS.get(results.param_names[idx], results.param_names[idx])
            s1 = results.s1[metric][idx]
            s1_c = results.s1_conf[metric][idx]
            st = results.st[metric][idx]
            st_c = results.st_conf[metric][idx]
            print(f"  {name:<22} | {s1:>6.3f} ({s1_c:>6.3f}) | {st:>6.3f} ({st_c:>6.3f}) |")

    # Second-order interactions (top 5 per metric)
    if results.s2 is not None:
        print(f"\n  --- Top Pairwise Interactions (S2) ---")
        for metric in results.metric_names:
            label = metric_labels.get(metric, metric)
            s2 = results.s2[metric]
            s2_conf = results.s2_conf[metric]
            D = len(results.param_names)

            # Collect all pairs
            pairs = []
            for i in range(D):
                for j in range(i + 1, D):
                    pairs.append((i, j, s2[i, j], s2_conf[i, j]))

            # Sort by absolute S2 descending
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            print(f"\n  {label}:")
            for i, j, s2_val, s2_c in pairs[:5]:
                n1 = PARAM_LABELS.get(results.param_names[i], results.param_names[i])
                n2 = PARAM_LABELS.get(results.param_names[j], results.param_names[j])
                sig = "*" if abs(s2_val) > s2_c else ""
                print(f"    {n1} x {n2}: S2={s2_val:>7.3f} (conf {s2_c:.3f}){sig}")

    # Percentile ranges from raw outputs
    if results.outputs is not None:
        print(f"\n  --- Output Distribution (from {len(results.outputs['peak_delay_wk'])} samples) ---")
        for metric in results.metric_names:
            label = metric_labels.get(metric, metric)
            data = results.outputs[metric]
            p10 = np.percentile(data, 10)
            p50 = np.percentile(data, 50)
            p90 = np.percentile(data, 90)
            unit = "wk" if "delay" in metric else "%"
            print(f"    {label}: p10={p10:.1f}{unit}, median={p50:.1f}{unit}, p90={p90:.1f}{unit}")

    print()
    print("  S1 = first-order (direct effect); ST = total-order (including interactions)")
    print("  S2 = pairwise interaction (* = significant: |S2| > confidence interval)")
    print("  Parameters with ST > 0.1 are meaningfully influential.")
    print()
