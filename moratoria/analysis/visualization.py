"""
Visualization module for the Data Center Moratoria Impact Model.

Produces publication-quality charts for geographic displacement,
capacity dynamics, and hardware compute shortfalls.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from moratoria.config import quarter_label, T_END, HARDWARE_MILESTONES
from moratoria.data.regions import blue_regions, red_regions, swing_regions, international_regions
from moratoria.simulation.runner import SimulationResults


# Style configuration
COLORS = {
    "baseline": "#2196F3",
    "currently_considering": "#FF9800",
    "all_dem_trifectas": "#F44336",
}

SCENARIO_LABELS = {
    "baseline": "Baseline (No Moratoria)",
    "currently_considering": "Currently Considering",
    "all_dem_trifectas": "All Dem Trifectas (excl. VA)",
}


def _quarter_labels(t_end: int, step: int = 4) -> tuple[list[int], list[str]]:
    """Generate quarterly tick positions and labels."""
    positions = list(range(0, t_end, step))
    labels = [quarter_label(t) for t in positions]
    return positions, labels


def plot_capacity_trajectory(
    results: dict[str, SimulationResults],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 7),
):
    """Total data center capacity (GW) across scenarios."""
    fig, ax = plt.subplots(figsize=figsize)

    for name, r in results.items():
        color = COLORS.get(name, "#666666")
        label = SCENARIO_LABELS.get(name, name)
        linewidth = 2.5 if name == "baseline" else 1.5
        linestyle = "-" if name == "baseline" else "--"

        ax.plot(
            r.capacity_trajectory / 1000,
            label=label, color=color,
            linewidth=linewidth, linestyle=linestyle,
        )

    positions, labels = _quarter_labels(len(list(results.values())[0].capacity_trajectory))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Total DC Capacity (GW)", fontsize=12)
    ax.set_title("Data Center Capacity: Baseline vs. Moratorium Scenarios",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_capacity_shortfall(
    results: dict[str, SimulationResults],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 7),
):
    """Capacity shortfall (%) relative to baseline."""
    fig, ax = plt.subplots(figsize=figsize)

    for name, r in results.items():
        if name == "baseline":
            continue
        if r.ai_timeline.capacity_shortfall_pct is None:
            continue

        color = COLORS.get(name, "#666666")
        label = SCENARIO_LABELS.get(name, name)

        ax.plot(
            r.ai_timeline.capacity_shortfall_pct,
            label=label, color=color, linewidth=1.5,
        )

    positions, labels = _quarter_labels(len(list(results.values())[0].capacity_trajectory))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Capacity Shortfall (%)", fontsize=12)
    ax.set_title("Capacity Shortfall vs. Baseline (the 'Delay Tax')",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_geographic_shift(
    result: SimulationResults,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """Stacked area chart showing capacity by political lean over time."""
    if result.capacity_by_region is None:
        return None

    fig, ax = plt.subplots(figsize=figsize)

    blue = blue_regions()
    red = red_regions()
    swing = swing_regions()
    intl = international_regions()

    t_end = len(result.capacity_trajectory)
    t_range = np.arange(t_end)

    blue_total = sum(result.capacity_by_region.get(r, np.zeros(t_end)) for r in blue) / 1000
    red_total = sum(result.capacity_by_region.get(r, np.zeros(t_end)) for r in red) / 1000
    swing_total = sum(result.capacity_by_region.get(r, np.zeros(t_end)) for r in swing) / 1000
    intl_total = sum(result.capacity_by_region.get(r, np.zeros(t_end)) for r in intl) / 1000

    ax.stackplot(
        t_range, blue_total, swing_total, red_total, intl_total,
        labels=["Blue States", "Swing States", "Red States", "International"],
        colors=["#2196F3", "#9C27B0", "#F44336", "#FF9800"],
        alpha=0.7,
    )

    positions, labels = _quarter_labels(t_end)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Installed Capacity (GW)", fontsize=12)
    ax.set_title(f"Geographic Distribution: {result.scenario_name}",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hardware_compute_trajectory(
    results: dict[str, SimulationResults],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 7),
):
    """Hardware training FLOP trajectory across scenarios (log scale)."""
    fig, ax = plt.subplots(figsize=figsize)

    for name, r in results.items():
        color = COLORS.get(name, "#666666")
        label = SCENARIO_LABELS.get(name, name)
        linewidth = 2.5 if name == "baseline" else 1.5
        linestyle = "-" if name == "baseline" else "--"

        ax.semilogy(
            r.ai_timeline.training_flops_trajectory,
            label=label, color=color,
            linewidth=linewidth, linestyle=linestyle,
        )

    # Add milestone lines
    for m_name, threshold in HARDWARE_MILESTONES.items():
        ax.axhline(y=threshold, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.text(0.5, threshold * 1.3, m_name.replace("_", " "),
                fontsize=8, color="gray", alpha=0.7)

    positions, labels = _quarter_labels(len(list(results.values())[0].capacity_trajectory))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("Hardware FLOP per Frontier Training Run", fontsize=12)
    ax.set_title("Hardware Compute for Frontier Training: Baseline vs. Moratoria",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_all_charts(
    scenario_results: dict[str, SimulationResults],
    output_dir: str = ".",
):
    """Generate all standard charts and save to output directory."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    plot_capacity_trajectory(
        scenario_results,
        save_path=os.path.join(output_dir, "01_capacity_trajectory.png"),
    )

    plot_capacity_shortfall(
        scenario_results,
        save_path=os.path.join(output_dir, "02_capacity_shortfall.png"),
    )

    plot_hardware_compute_trajectory(
        scenario_results,
        save_path=os.path.join(output_dir, "03_hardware_compute_trajectory.png"),
    )

    # Geographic shift for each non-baseline scenario
    for name, r in scenario_results.items():
        if name == "baseline":
            continue
        plot_geographic_shift(
            r,
            save_path=os.path.join(output_dir, f"04_geographic_shift_{name}.png"),
        )

    plt.close("all")
