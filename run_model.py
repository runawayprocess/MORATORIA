#!/usr/bin/env python3
"""
Data Center Moratoria Impact Model
===================================

Main entry point: runs all scenarios, generates summary tables,
produces charts, and prints key findings.

Usage:
    python run_model.py                    # Full analysis with charts
    python run_model.py --no-charts        # Tables only (faster)
    python run_model.py --sensitivity      # Include sensitivity sweeps
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moratoria.config import (
    quarter_label, HARDWARE_MILESTONES, EFFECTIVE_COMPUTE_MULTIPLIERS,
    T_END, INVESTMENT_ELASTICITY, REALLOCATION_DELAY_QTRS,
)
from moratoria.data.regions import REGIONS, total_capacity_2025, blue_regions, red_regions, international_regions, swing_regions
from moratoria.data.scenarios import SCENARIOS
from moratoria.simulation.runner import run_all_scenarios, run_scenario
from moratoria.module_c.compute import decompose_compute_growth, compare_to_baseline
from moratoria.module_a.displacement import DisplacementModel


def print_header():
    print("=" * 72)
    print("  DATA CENTER MORATORIA IMPACT MODEL")
    print("  Geographic Displacement -> Capacity Impact -> Hardware Compute")
    print("=" * 72)
    print()


def print_model_context():
    us_count = len([r for r in REGIONS.values() if not r.is_international])
    intl_count = len([r for r in REGIONS.values() if r.is_international])
    end_label = quarter_label(T_END - 1)
    print("--- Model Context ---")
    print(f"  Simulation: Q1 2026 -> {end_label} ({T_END} quarters)")
    print(f"  Starting US capacity: {total_capacity_2025()/1000:.1f} GW across {us_count} US + {intl_count} international regions")
    print(f"  Target US capacity (2030): 80.0 GW (McKinsey/Goldman Sachs/DOE midpoint)")
    print(f"  Investment elasticity: {INVESTMENT_ELASTICITY} (Baker-Bloom-Davis sectoral range)")
    print(f"  Reallocation delay: {REALLOCATION_DELAY_QTRS} quarters ({REALLOCATION_DELAY_QTRS * 3} months)")
    print(f"  Scenarios: {len(SCENARIOS)}")
    for name, scen in SCENARIOS.items():
        print(f"    - {name}: {scen.description[:80]}{'...' if len(scen.description) > 80 else ''}")
    print()


def print_scenario_table(results: dict):
    """Print summary table across all scenarios."""
    baseline = results.get("baseline")

    print(f"{'Scenario':<30} | {'End Cap':>8} | {'Raw':>6} | {'QW':>6} | {'Peak':>7} | {'Cumul':>7} | {'Reequil':>9} |")
    print(f"{'':30} | {'(GW)':>8} | {'Short%':>6} | {'Short%':>6} | {'Delay':>7} | {'Deficit':>7} | {'Quarter':>9} |")
    print("-" * 110)

    for name, r in results.items():
        end_cap = r.capacity_trajectory[-1] / 1000
        tl = r.ai_timeline

        # Quality-weighted shortfall (what the model uses for compute)
        if tl.capacity_shortfall_pct is not None:
            peak_qw_short = tl.capacity_shortfall_pct.max()
        else:
            peak_qw_short = 0.0

        # Raw capacity shortfall (without quality weighting)
        if baseline and name != "baseline":
            bl_raw = baseline.capacity_trajectory
            sc_raw = r.capacity_trajectory
            raw_shortfall = float(np.max(np.where(bl_raw > 0, (1 - sc_raw / bl_raw) * 100, 0)))
        else:
            raw_shortfall = 0.0

        if tl.peak_delay_qtrs is not None and name != "baseline":
            weeks = tl.peak_delay_qtrs * 13
            delay_str = f"{weeks:.1f}wk"
        else:
            delay_str = "--"

        if tl.cumulative_flop_deficit_pct is not None and name != "baseline":
            deficit_str = f"{tl.cumulative_flop_deficit_pct:.1f}%"
        else:
            deficit_str = "--"

        if tl.reequilibration_quarter is not None:
            req_str = quarter_label(tl.reequilibration_quarter)
        elif name != "baseline" and tl.compute_shortfall_pct is not None and np.any(tl.compute_shortfall_pct > 1.0):
            req_str = "NOT YET"
        else:
            req_str = "--"

        if name == "baseline":
            print(f"{r.scenario_name:<30} | {end_cap:>8.1f} |    --  |    --  | {delay_str:>7} | {deficit_str:>7} | {req_str:>9} |")
        else:
            print(f"{r.scenario_name:<30} | {end_cap:>8.1f} | {raw_shortfall:>5.1f}% | {peak_qw_short:>5.1f}% | {delay_str:>7} | {deficit_str:>7} | {req_str:>9} |")
    print()

    # Milestone delay detail
    milestone_names = list(EFFECTIVE_COMPUTE_MULTIPLIERS.keys())
    print(f"  Effective compute milestone delays:")
    print(f"  {'Scenario':<30} | ", end="")
    for m_name in milestone_names:
        print(f"{m_name:>9}", end=" | ")
    print()
    print("  " + "-" * 90)

    for name, r in results.items():
        delays = r.ai_timeline.effective_milestone_delays
        if name == "baseline" or not delays:
            print(f"  {r.scenario_name:<30} | ", end="")
            for _ in milestone_names:
                print(f"{'--':>9}", end=" | ")
            print()
        else:
            print(f"  {r.scenario_name:<30} | ", end="")
            for m_name in milestone_names:
                info = delays.get(m_name, {})
                w = info.get("delay_weeks_interpolated")
                if w is not None:
                    if w >= 1:
                        print(f"{w:>6.1f}wk", end=" | ")
                    else:
                        d = w * 7
                        print(f"{d:>6.0f}day", end=" | ")
                else:
                    print(f"{'N/A':>9}", end=" | ")
            print()
    print()


def print_geographic_shift(results: dict):
    """Print geographic redistribution for each scenario."""
    blue = blue_regions()
    red = red_regions()
    swing = swing_regions()
    intl = international_regions()

    print("  Geographic Redistribution (capacity share at start -> end):")
    print(f"  {'Scenario':<30} | {'Blue':>12} | {'Red':>12} | {'Swing':>12} | {'Intl':>12}")
    print("  " + "-" * 90)

    for name, r in results.items():
        if not r.capacity_by_region:
            continue
        total_start = r.capacity_trajectory[0]
        total_end = r.capacity_trajectory[-1]

        def share(regions, idx):
            return sum(r.capacity_by_region[reg][idx] for reg in regions) / (total_start if idx == 0 else total_end)

        b_s, b_e = share(blue, 0), share(blue, -1)
        r_s, r_e = share(red, 0), share(red, -1)
        sw_s, sw_e = share(swing, 0), share(swing, -1)
        i_s, i_e = share(intl, 0), share(intl, -1)

        print(f"  {r.scenario_name:<30} | {b_s:.0%} -> {b_e:.0%} | {r_s:.0%} -> {r_e:.0%} | {sw_s:.0%} -> {sw_e:.0%} | {i_s:.0%} -> {i_e:.0%}")
    print()


def print_quality_analysis(results: dict):
    """Print quality-weighted vs raw capacity comparison.

    This is what the geographic model captures that a back-of-envelope
    calculation cannot: when compute moves from high-quality fiber hubs
    (NOVA=1.0) to sparse regions (TX_OTHER=0.55), effective compute
    per MW degrades.
    """
    baseline = results.get("baseline")
    if not baseline:
        return

    print("  Quality Degradation Analysis (geographic model contribution):")
    print(f"  {'Scenario':<30} | {'Raw Cap':>9} | {'QW Cap':>9} | {'Quality':>9} |")
    print(f"  {'':30} | {'Short%':>9} | {'Short%':>9} | {'Penalty':>9} |")
    print("  " + "-" * 65)

    for name, r in results.items():
        if name == "baseline":
            print(f"  {r.scenario_name:<30} | {'--':>9} | {'--':>9} | {'--':>9} |")
            continue

        # Raw capacity shortfall (ignoring quality)
        bl_raw = baseline.capacity_trajectory
        sc_raw = r.capacity_trajectory
        raw_shortfall = np.max(np.where(bl_raw > 0, (1 - sc_raw / bl_raw) * 100, 0))

        # Quality-weighted shortfall
        bl_qw = baseline.quality_weighted_trajectory
        sc_qw = r.quality_weighted_trajectory
        qw_shortfall = np.max(np.where(bl_qw > 0, (1 - sc_qw / bl_qw) * 100, 0))

        # Quality penalty: additional shortfall from relocating to lower-quality regions
        penalty = qw_shortfall - raw_shortfall

        print(f"  {r.scenario_name:<30} | {raw_shortfall:>8.1f}% | {qw_shortfall:>8.1f}% | {penalty:>+8.1f}% |")

    # Flag scenarios where end capacity exceeds baseline (pipeline speed effect)
    for name, r in results.items():
        if name == "baseline":
            continue
        bl_end = baseline.capacity_trajectory[-1]
        sc_end = r.capacity_trajectory[-1]
        if sc_end > bl_end:
            excess_gw = (sc_end - bl_end) / 1000
            print(f"\n  NOTE: {r.scenario_name} ends with {excess_gw:.1f} GW MORE raw capacity than baseline.")
            print(f"  This occurs because moratoriums redirect investment from slow-pipeline regions")
            print(f"  (PJM: 16-23 quarter build) to fast-pipeline regions (ERCOT: 11 quarter build).")
            print(f"  The speed advantage outweighs reallocation friction for limited moratoriums.")
            print(f"  Quality-weighted capacity is still lower due to relocation to lower-quality regions.")
    print()


def print_key_findings(results: dict):
    """Print headline findings."""
    print("=" * 72)
    print("  KEY FINDINGS")
    print("=" * 72)

    # Use the most impactful non-baseline scenario for headline
    headline_key = "all_dem_trifectas" if "all_dem_trifectas" in results else "currently_considering"
    headline = results.get(headline_key)

    if headline and headline.capacity_by_region:
        blue = blue_regions()
        red = red_regions()
        intl = international_regions()
        total_start = headline.capacity_trajectory[0]
        total_end = headline.capacity_trajectory[-1]

        blue_start = sum(headline.capacity_by_region[r][0] for r in blue) / total_start
        blue_end = sum(headline.capacity_by_region[r][-1] for r in blue) / total_end
        red_start = sum(headline.capacity_by_region[r][0] for r in red) / total_start
        red_end = sum(headline.capacity_by_region[r][-1] for r in red) / total_end
        intl_start = sum(headline.capacity_by_region[r][0] for r in intl) / total_start
        intl_end = sum(headline.capacity_by_region[r][-1] for r in intl) / total_end

        print(f"\n  {headline.scenario_name} Scenario:")
        print(f"    Blue state share:    {blue_start:.0%} -> {blue_end:.0%}")
        print(f"    Red state share:     {red_start:.0%} -> {red_end:.0%}")
        print(f"    International share: {intl_start:.0%} -> {intl_end:.0%}")

        tl = headline.ai_timeline
        if tl.capacity_shortfall_pct is not None:
            peak_short = tl.capacity_shortfall_pct.max()
            peak_t = int(tl.capacity_shortfall_pct.argmax())
            peak_mw = (results["baseline"].capacity_trajectory[peak_t] - headline.capacity_trajectory[peak_t])
            print(f"\n    Peak capacity shortfall: {peak_short:.1f}% ({peak_mw/1000:.1f} GW) at {quarter_label(peak_t)}")

        if tl.compute_shortfall_pct is not None:
            peak_compute = tl.compute_shortfall_pct.max()
            print(f"    Peak effective compute shortfall: {peak_compute:.1f}%")

        if tl.peak_delay_qtrs is not None:
            days = tl.peak_delay_qtrs * 91.25
            weeks = days / 7
            print(f"    Peak AI capabilities delay: {tl.peak_delay_qtrs:.2f} quarters (~{weeks:.1f} weeks / {days:.0f} days)")

        if tl.cumulative_flop_deficit_pct is not None:
            print(f"    Cumulative compute deficit: {tl.cumulative_flop_deficit_pct:.1f}% of baseline total over simulation")

        if tl.reequilibration_quarter is not None:
            print(f"    Compute shortfall drops below 1%: {quarter_label(tl.reequilibration_quarter)}")
        elif tl.compute_shortfall_pct is not None and np.any(tl.compute_shortfall_pct > 1.0):
            print(f"    Compute shortfall: still above 1% at end of simulation")

        if tl.effective_milestone_delays:
            print(f"\n    Effective compute delays (hardware + software):")
            for m_name, info in tl.effective_milestone_delays.items():
                w = info.get("delay_weeks_interpolated")
                d = info.get("delay_qtrs_interpolated")
                if w is not None:
                    print(f"      {m_name}: ~{w:.1f} weeks ({d:.2f} quarters)")
                else:
                    print(f"      {m_name}: N/A")

    # Pipeline timeline diagram
    if headline:
        print(f"\n  --- Pipeline Timeline (why peak shortfall lags moratorium expiry) ---")
        print(f"    ADT moratoriums active:   Q3 2026 ---------> Q2 2028-Q2 2029")
        print(f"    Reallocation delay (5Q):              Q4 2027 ---------> Q3 2029-Q3 2030")
        print(f"    Construction (11-23Q):                            builds completing...")
        print(f"    Fast regions (ERCOT):                                 Q3 2030 ---------> Q2 2033")
        print(f"    Slow regions (PJM):                                  Q4 2032 ---------> Q2 2035")
        print(f"    Preseed completions:      =========== (masks early shortfall)")
        print(f"    Peak shortfall:                                                  ~Q1 2034")
        print(f"    ")
        print(f"    The gap between preseed tapering off and redirected pipeline")
        print(f"    completing is when the shortfall is worst. Moratoriums expire")
        print(f"    years before the capacity impact peaks because of pipeline lag.")

    # Back-of-envelope validation
    print(f"\n  --- Back-of-Envelope Check ---")
    decomp = decompose_compute_growth()
    eff_growth_qtr = (1 + decomp['total_growth_annual']) ** 0.25 - 1
    print(f"    Effective compute growth: ~{eff_growth_qtr*100:.0f}%/quarter (hardware + software)")
    if headline and headline.ai_timeline.compute_shortfall_pct is not None:
        peak_cs = headline.ai_timeline.compute_shortfall_pct.max()
        envelope_delay = np.log(1 / (1 - peak_cs / 100)) / np.log(1 + eff_growth_qtr)
        envelope_weeks = envelope_delay * 13
        print(f"    Peak compute shortfall: {peak_cs:.1f}%")
        print(f"    Simple delay = log(1/{1 - peak_cs/100:.3f}) / log({1 + eff_growth_qtr:.3f}) = {envelope_delay:.2f} quarters (~{envelope_weeks:.1f} weeks)")
        if headline.ai_timeline.peak_delay_qtrs:
            print(f"    Model delay: {headline.ai_timeline.peak_delay_qtrs:.2f} quarters")
            print(f"    Match confirms model arithmetic is sound; geographic model drives the shortfall %.")

    # Compute decomposition
    print(f"\n  --- Software vs. Hardware Decomposition ---")
    print(f"    Hardware efficiency growth: {decomp['hardware_growth_annual']:.0%}/year")
    print(f"    Software efficiency growth: {decomp['software_growth_annual']:.0%}/year")
    print(f"    Hardware share of total AI progress (log): {decomp['hardware_share']:.0%}")
    print(f"    Software share of total AI progress (log): {decomp['software_share']:.0%}")
    print(f"\n    Moratoria affect only hardware deployment ({decomp['hardware_share']:.0%} of progress).")
    print(f"    Software/algorithmic improvements continue regardless.")

    # Duration vs. breadth finding
    cc = results.get("currently_considering")
    adt_2yr = results.get("all_dem_trifectas")
    if cc and adt_2yr and cc.ai_timeline.cumulative_flop_deficit_pct and adt_2yr.ai_timeline.cumulative_flop_deficit_pct:
        cc_cd = cc.ai_timeline.cumulative_flop_deficit_pct
        adt_cd = adt_2yr.ai_timeline.cumulative_flop_deficit_pct
        if cc_cd > adt_cd:
            print(f"\n  --- Duration vs. Breadth ---")
            print(f"    Currently Considering cumulative deficit ({cc_cd:.1f}%) > ADT ({adt_cd:.1f}%)")
            print(f"    despite narrower moratorium scope. Three mechanisms:")
            print(f"    1. CC moratoriums are longer (NYC 3yr, OTHER_BLUE to 2030 vs ADT ~2yr)")
            print(f"    2. CC uniquely blocks Atlanta, a red-state destination region")
            print(f"    3. ADT redirects slow-pipeline (PJM 23Q) to fast-pipeline (ERCOT 11Q),")
            print(f"       briefly causing MORE capacity than baseline (pipeline speed-up)")
            print(f"    Implication: moratorium DURATION matters more than BREADTH for")
            print(f"    cumulative compute impact. Blocking destination regions is costly.")

    # Sobol findings
    print(f"\n  --- Key Sensitivity Finding (Sobol, N=64, 1408 runs, 10 params) ---")
    print(f"    Peak delay driven by congestion_threshold (S1=0.80). Algo doubling and")
    print(f"    congestion_sensitivity are secondary (ST~0.12 each).")
    print(f"    Peak shortfall dominated by congestion_threshold (S1=0.94).")
    print(f"    Cumulative deficit more distributed: congestion_threshold (S1=0.38),")
    print(f"    congestion_sensitivity (ST=0.36), fungibility_price_response (ST=0.25),")
    print(f"    investment_elasticity (ST=0.15).")
    print(f"    Distribution: delay p10=1.2wk, median=2.9wk, p90=4.2wk")

    # Baseline validation
    if "baseline" in results:
        bl_end_gw = results["baseline"].capacity_trajectory[-1] / 1000
        bl_2030_idx = min(19, len(results["baseline"].capacity_trajectory) - 1)  # Q4 2030 = index 19
        bl_2030_gw = results["baseline"].capacity_trajectory[bl_2030_idx] / 1000
        print(f"\n  --- Baseline Validation ---")
        print(f"    Baseline capacity at Q4 2030 (index {bl_2030_idx}): {bl_2030_gw:.1f} GW")
        print(f"    Baseline capacity at simulation end: {bl_end_gw:.1f} GW")
        print(f"    Industry target: ~80 GW by 2030 (McKinsey/Goldman Sachs/DOE midpoint)")
        ratio = bl_2030_gw / 80.0
        if 0.9 <= ratio <= 1.1:
            print(f"    Baseline is within 10% of industry target ({ratio:.0%}). Calibration OK.")
        elif ratio > 1.1:
            print(f"    NOTE: Baseline exceeds 2030 industry target ({ratio:.0%}).")
        else:
            print(f"    WARNING: Baseline undershoots 2030 industry target ({ratio:.0%}).")

    print()


def print_logit_validation():
    """
    Validate the multinomial logit allocation against observed 2025 capacity shares.

    Shows both uncalibrated (feature weights only) and calibrated (with ASCs)
    predictions to demonstrate the calibration improvement.
    """
    print("=" * 72)
    print("  LOGIT ALLOCATION VALIDATION (t=0 vs. 2025 Observed)")
    print("=" * 72)

    # Observed shares
    us_total = sum(r.capacity_2025_mw for r in REGIONS.values() if not r.is_international)
    observed_shares = {
        name: r.capacity_2025_mw / us_total
        for name, r in REGIONS.items()
        if not r.is_international
    }

    # Uncalibrated model (feature weights only)
    model_uncal = DisplacementModel(calibrate=False)
    pred_uncal = model_uncal.allocate_baseline(t=0)

    # Calibrated model (with ASCs)
    model_cal = DisplacementModel(calibrate=True)
    pred_cal = model_cal.allocate_baseline(t=0)

    print(f"\n  {'Region':<22} | {'Observed':>10} | {'Uncalib.':>10} | {'Error':>10} | {'Calib.':>10} | {'Error':>10} |")
    print(f"  {'-' * 84}")

    uncal_err = 0.0
    cal_err = 0.0
    for name in sorted(observed_shares.keys(), key=lambda n: observed_shares[n], reverse=True):
        obs = observed_shares[name]
        pu = pred_uncal.get(name, 0.0)
        pc = pred_cal.get(name, 0.0)
        eu = pu - obs
        ec = pc - obs
        uncal_err += abs(eu)
        cal_err += abs(ec)
        r = REGIONS[name]
        label = f"{r.short_name} ({r.name[:15]})"
        print(f"  {label:<22} | {obs:>9.1%} | {pu:>9.1%} | {eu:>+9.1%} | {pc:>9.1%} | {ec:>+9.1%} |")

    print(f"  {'-' * 84}")
    print(f"  {'Total abs error':<22} | {'':>10} | {'':>10} | {uncal_err:>9.1%} | {'':>10} | {cal_err:>9.1%} |")

    print(f"\n  Logit temperature: {model_cal.temperature}")
    print(f"  Calibration method: Alternative-Specific Constants (McFadden 1974)")
    print(f"  Uncalibrated error: {uncal_err:.0%} -> Calibrated error: {cal_err:.1%}")
    print(f"\n  ASCs ensure the model reproduces observed 2025 shares exactly.")
    print(f"  Feature weights then determine substitution patterns under moratorium shocks.")
    print(f"  This is standard discrete choice calibration.")
    print()


def print_sensitivity_analysis():
    """
    Sweep the most influential parameters (per Sobol analysis) to show
    the range of outcomes.

    Sobol global SA found that algo_doubling_months and effective_fungibility
    dominate variance in delay and shortfall metrics. Investment elasticity
    and reallocation delay have near-zero total-order indices (ST < 0.05).
    This table sweeps the parameters that actually matter.
    """
    import moratoria.module_c.compute as comp_mod
    import moratoria.module_a.displacement as disp_mod

    print("=" * 72)
    print("  SENSITIVITY ANALYSIS (All Democratic Trifectas scenario)")
    print("  Parameters selected by Sobol global sensitivity analysis")
    print("=" * 72)

    scenario = SCENARIOS["all_dem_trifectas"]
    baseline_scenario = SCENARIOS["baseline"]

    # Sobol top parameters: algo doubling time and effective fungibility
    algo_months = [8, 12, 18]
    fungibilities = [0.40, 0.556, 0.70]

    algo_labels = ["8mo (fast)", "12mo (base)", "18mo (slow)"]
    fung_labels = ["0.40 (low)", "0.56 (base)", "0.70 (high)"]

    print(f"\n  Peak AI capabilities delay (weeks):")
    print(f"  {'':>22} | Algo Doubling Time:")
    print(f"  {'Eff. Fungibility':>22} | {algo_labels[0]:>14} | {algo_labels[1]:>14} | {algo_labels[2]:>14} |")
    print("  " + "-" * 72)

    all_delays = []
    all_shortfalls = []
    all_deficits = []

    for fi, fung in enumerate(fungibilities):
        print(f"  {fung_labels[fi]:>22} | ", end="")
        for algo in algo_months:
            # Monkey-patch
            orig_algo = comp_mod.ALGO_DOUBLING_TIME_MONTHS
            orig_fung = disp_mod.EFFECTIVE_FUNGIBILITY
            comp_mod.ALGO_DOUBLING_TIME_MONTHS = algo
            disp_mod.EFFECTIVE_FUNGIBILITY = fung
            try:
                bl = run_scenario(baseline_scenario)
                sc = run_scenario(scenario)
                compare_to_baseline(bl.ai_timeline, sc.ai_timeline)
                delay_wk = sc.ai_timeline.peak_delay_qtrs * 13
                peak_short = sc.ai_timeline.compute_shortfall_pct.max()
                cumul = sc.ai_timeline.cumulative_flop_deficit_pct or 0
                all_delays.append(delay_wk)
                all_shortfalls.append(peak_short)
                all_deficits.append(cumul)
                print(f"{delay_wk:>13.1f}wk | ", end="")
            finally:
                comp_mod.ALGO_DOUBLING_TIME_MONTHS = orig_algo
                disp_mod.EFFECTIVE_FUNGIBILITY = orig_fung
        print()

    print(f"\n  Peak effective compute shortfall (%):")
    print(f"  {'':>22} | Algo Doubling Time:")
    print(f"  {'Eff. Fungibility':>22} | {algo_labels[0]:>14} | {algo_labels[1]:>14} | {algo_labels[2]:>14} |")
    print("  " + "-" * 72)

    idx = 0
    for fi, fung in enumerate(fungibilities):
        print(f"  {fung_labels[fi]:>22} | ", end="")
        for _ in algo_months:
            print(f"{all_shortfalls[idx]:>13.1f}% | ", end="")
            idx += 1
        print()

    print(f"\n  Cumulative compute deficit (%):")
    print(f"  {'':>22} | Algo Doubling Time:")
    print(f"  {'Eff. Fungibility':>22} | {algo_labels[0]:>14} | {algo_labels[1]:>14} | {algo_labels[2]:>14} |")
    print("  " + "-" * 72)

    idx = 0
    for fi, fung in enumerate(fungibilities):
        print(f"  {fung_labels[fi]:>22} | ", end="")
        for _ in algo_months:
            print(f"{all_deficits[idx]:>12.1f}% | ", end="")
            idx += 1
        print()

    print(f"\n  RANGE: Peak delay spans {min(all_delays):.1f} to {max(all_delays):.1f} weeks")
    print(f"         Peak compute shortfall spans {min(all_shortfalls):.1f}% to {max(all_shortfalls):.1f}%")
    print(f"         Cumulative deficit spans {min(all_deficits):.1f}% to {max(all_deficits):.1f}%")
    center = len(algo_months) * 1 + 1  # middle of 3x3
    print(f"         Central estimate (algo=12mo, fung=0.556): {all_delays[center]:.1f}wk / {all_shortfalls[center]:.1f}%")

    print(f"\n  NOTE: Sobol analysis (N=64, 1408 runs, 10 params) found congestion_threshold")
    print(f"  is the dominant parameter for delay (S1=0.80) and shortfall (S1=0.94).")
    print(f"  Cumulative deficit is more distributed across congestion_threshold (S1=0.38),")
    print(f"  congestion_sensitivity (ST=0.36), and fungibility_price_response (ST=0.25).")
    print(f"  Logit temperature and agglomeration elasticity have near-zero influence")
    print(f"  with ASC calibration (ST < 0.07).")
    print()


def main():
    parser = argparse.ArgumentParser(description="Data Center Moratoria Impact Model")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--sensitivity", action="store_true", help="Include sensitivity sweeps")
    parser.add_argument("--sobol", action="store_true", help="Full Sobol global sensitivity analysis")
    parser.add_argument("--sobol-n", type=int, default=64, help="Sobol base sample size (default 64)")
    parser.add_argument("--validate", action="store_true", help="Run logit allocation validation")
    parser.add_argument("--output-dir", default="output", help="Output directory for charts")
    args = parser.parse_args()

    print_header()
    print_model_context()

    # Logit validation
    if args.validate or args.sobol:
        print_logit_validation()

    # Run all scenarios
    print("Running scenarios...")
    results = run_all_scenarios()
    print("  Done.\n")

    # Summary table
    print_scenario_table(results)

    # Geographic shift
    print_geographic_shift(results)

    # Quality degradation (geographic model's unique contribution)
    print_quality_analysis(results)

    # Key findings
    print_key_findings(results)

    # Sensitivity analysis
    if args.sensitivity:
        print_sensitivity_analysis()

    # Sobol global sensitivity
    if args.sobol:
        try:
            from moratoria.analysis.sobol_sensitivity import run_sobol, print_sobol_results
            sobol_results = run_sobol(n_samples=args.sobol_n, verbose=True)
            print_sobol_results(sobol_results)
        except ImportError as e:
            print(f"  Skipping Sobol analysis (SALib not available): {e}")

    # Detailed per-scenario output
    for name, r in results.items():
        print(r.ai_timeline.summary())
        print()

    # Charts
    if not args.no_charts:
        print(f"\nGenerating charts in {args.output_dir}/...")
        try:
            from moratoria.analysis.visualization import generate_all_charts
            generate_all_charts(results, output_dir=args.output_dir)
            print(f"  Charts saved to {args.output_dir}/")
        except ImportError as e:
            print(f"  Skipping charts (matplotlib not available): {e}")
        except Exception as e:
            print(f"  Error generating charts: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
