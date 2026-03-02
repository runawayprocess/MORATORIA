"""
Reporting utilities for the moratoria impact model.

Includes scenario summary table generation and compute growth decomposition.
"""

import numpy as np

from moratoria.config import HARDWARE_MILESTONES, quarter_label
from moratoria.module_c.compute import decompose_compute_growth
from moratoria.simulation.runner import SimulationResults


def generate_scenario_summary_table(
    results: dict[str, SimulationResults],
) -> str:
    """Generate a formatted summary table of all scenario results."""
    milestone_names = list(HARDWARE_MILESTONES.keys())
    header = (
        f"{'Scenario':<28} | {'Peak Cap (GW)':>13} | {'Peak Short%':>11} | "
        + " | ".join(f"{n.replace('_hardware', 'x'):>8}" for n in milestone_names)
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for name, r in results.items():
        peak_cap = r.capacity_trajectory.max() / 1000 if r.capacity_trajectory is not None else 0
        peak_short = r.ai_timeline.capacity_shortfall_pct.max() if r.ai_timeline.capacity_shortfall_pct is not None else 0

        delays = []
        for m_name in milestone_names:
            if r.ai_timeline.milestone_delays and m_name in r.ai_timeline.milestone_delays:
                d = r.ai_timeline.milestone_delays[m_name].get("delay_qtrs_interpolated")
                delays.append(f"{d:.1f}q" if d is not None else "N/R")
            else:
                delays.append("--")

        row = f"{name:<28} | {peak_cap:>13.1f} | {peak_short:>10.1f}% | " + " | ".join(f"{d:>8}" for d in delays)
        rows.append(row)

    return "\n".join(rows)


def print_compute_decomposition(
    algo_doubling_months: float = 12,
    hardware_growth_annual: float = 0.30,
) -> str:
    """Print the hardware vs. software decomposition of compute growth."""
    decomp = decompose_compute_growth(algo_doubling_months, hardware_growth_annual)
    lines = [
        "\n=== Compute Growth Decomposition ===",
        f"Hardware efficiency growth:  {decomp['hardware_growth_annual']:.0%}/year",
        f"Software efficiency growth:  {decomp['software_growth_annual']:.0%}/year",
        f"Total effective compute growth: {decomp['total_growth_annual']:.0%}/year",
        f"",
        f"Hardware share of growth (log): {decomp['hardware_share']:.1%}",
        f"Software share of growth (log): {decomp['software_share']:.1%}",
        f"",
        f"Implication: A moratorium causing X quarters of hardware delay",
        f"translates to ~{decomp['effective_delay_fraction']:.1%} * X quarters",
        f"of effective AI capabilities delay.",
    ]
    return "\n".join(lines)
