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

from moratoria.config import quarter_label, HARDWARE_MILESTONES
from moratoria.data.regions import REGIONS, total_capacity_2025, blue_regions, red_regions, international_regions, swing_regions
from moratoria.data.scenarios import SCENARIOS
from moratoria.simulation.runner import run_all_scenarios
from moratoria.module_c.compute import decompose_compute_growth


def print_header():
    print("=" * 72)
    print("  DATA CENTER MORATORIA IMPACT MODEL")
    print("  Geographic Displacement -> Capacity Impact -> Hardware Compute")
    print("=" * 72)
    print()


def print_model_context():
    us_count = len([r for r in REGIONS.values() if not r.is_international])
    intl_count = len([r for r in REGIONS.values() if r.is_international])
    print("--- Model Context ---")
    print(f"  Simulation: Q1 2026 -> Q4 2032 (28 quarters)")
    print(f"  Starting US capacity: {total_capacity_2025()/1000:.1f} GW across {us_count} US + {intl_count} international regions")
    print(f"  Target US capacity (2030): 55.0 GW")
    print(f"  Scenarios: {len(SCENARIOS)}")
    for name, scen in SCENARIOS.items():
        print(f"    - {name}: {scen.description[:70]}{'...' if len(scen.description) > 70 else ''}")
    print()


def print_scenario_table(results: dict):
    """Print summary table across all scenarios."""
    print(f"{'Scenario':<30} | {'Peak Cap (GW)':>13} | {'Peak Short%':>11} | ", end="")
    for m_name in HARDWARE_MILESTONES:
        label = m_name.replace("_hardware", "x")
        print(f"{label:>9}", end=" | ")
    print()
    print("-" * 110)

    baseline = results.get("baseline")
    for name, r in results.items():
        peak_cap = r.capacity_trajectory[-1] / 1000
        if r.ai_timeline.capacity_shortfall_pct is not None:
            peak_short = r.ai_timeline.capacity_shortfall_pct.max()
        else:
            peak_short = 0.0

        print(f"{r.scenario_name:<30} | {peak_cap:>13.1f} | {peak_short:>10.1f}% | ", end="")

        if name == "baseline" or not r.ai_timeline.milestone_delays:
            for _ in HARDWARE_MILESTONES:
                print(f"{'--':>9}", end=" | ")
        else:
            for m_name in HARDWARE_MILESTONES:
                info = r.ai_timeline.milestone_delays.get(m_name, {})
                d = info.get("delay_qtrs_interpolated")
                if d is not None:
                    print(f"{d:>8.1f}q", end=" | ")
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

        if headline.ai_timeline.capacity_shortfall_pct is not None:
            peak_short = headline.ai_timeline.capacity_shortfall_pct.max()
            peak_t = headline.ai_timeline.capacity_shortfall_pct.argmax()
            print(f"\n    Peak capacity shortfall: {peak_short:.1f}% at {quarter_label(peak_t)}")

        if headline.ai_timeline.compute_shortfall_pct is not None:
            peak_compute = headline.ai_timeline.compute_shortfall_pct.max()
            print(f"    Peak hardware compute shortfall: {peak_compute:.1f}%")

    # Compute decomposition (analytical, not simulated)
    decomp = decompose_compute_growth()
    print(f"\n  Software vs. Hardware (analytical context, not simulated):")
    print(f"    Hardware efficiency growth: {decomp['hardware_growth_annual']:.0%}/year")
    print(f"    Software efficiency growth: {decomp['software_growth_annual']:.0%}/year")
    print(f"    Hardware share of total AI progress (log): {decomp['hardware_share']:.0%}")
    print(f"    Software share of total AI progress (log): {decomp['software_share']:.0%}")
    print(f"\n    Implication: Moratoria affect only hardware deployment ({decomp['hardware_share']:.0%} of progress).")
    print(f"    Software/algorithmic improvements continue regardless of moratoria.")
    print(f"    A hardware delay of D quarters translates to ~{decomp['hardware_share']:.0%} * D quarters")
    print(f"    of effective AI capabilities delay.")

    print()


def main():
    parser = argparse.ArgumentParser(description="Data Center Moratoria Impact Model")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--sensitivity", action="store_true", help="Include sensitivity sweeps")
    parser.add_argument("--output-dir", default="output", help="Output directory for charts")
    args = parser.parse_args()

    print_header()
    print_model_context()

    # Run all scenarios
    print("Running scenarios...")
    results = run_all_scenarios()
    print("  Done.\n")

    # Summary table
    print_scenario_table(results)

    # Geographic shift
    print_geographic_shift(results)

    # Key findings
    print_key_findings(results)

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
