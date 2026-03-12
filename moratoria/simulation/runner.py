"""
Main simulation loop integrating Module A -> Module B -> Module C.

Runs a quarterly forward simulation from Q1 2026 through Q4 2032.
Each quarter:
  1. Module A allocates national investment across regions (with moratoria)
  2. Investment enters regional pipelines in Module B
  3. Module B processes completions and depreciation
  4. Agglomeration updates

After the time-series, Module C maps the quality-weighted capacity
trajectory to hardware compute milestones.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from moratoria.config import T_END, INVESTMENT_ELASTICITY, REALLOCATION_DELAY_QTRS, quarter_label
from moratoria.data.regions import REGIONS
from moratoria.data.scenarios import Scenario, SCENARIOS
from moratoria.module_a.displacement import DisplacementModel
from moratoria.module_b.capacity import CapacityModel
from moratoria.module_b.baseline import compute_baseline_investment_curve, compute_preseed_completions
from moratoria.module_c.compute import run_compute, compare_to_baseline, AITimelineResults


@dataclass
class SimulationResults:
    """Complete results from a full simulation run."""
    scenario_name: str
    scenario: Scenario
    t_end: int

    capacity_trajectory: Optional[np.ndarray] = None
    us_capacity_trajectory: Optional[np.ndarray] = None
    quality_weighted_trajectory: Optional[np.ndarray] = None
    capacity_by_region: Optional[dict] = None
    blocked_trajectory: Optional[np.ndarray] = None

    ai_timeline: Optional[AITimelineResults] = None


def run_scenario(
    scenario: Scenario,
    t_end: int = T_END,
    reallocation_delay: int = None,
    investment_elasticity: float = None,
) -> SimulationResults:
    """Run a full simulation for one scenario."""
    displacement = DisplacementModel(
        reallocation_delay=reallocation_delay or REALLOCATION_DELAY_QTRS,
    )
    inv_elasticity = investment_elasticity if investment_elasticity is not None else INVESTMENT_ELASTICITY
    capacity_model = CapacityModel()
    investment_curve = compute_baseline_investment_curve(t_end=t_end)

    # Pre-seeded pipeline: distributed by estimated 2025 pipeline geography,
    # NOT by installed capacity. The actual construction pipeline is
    # concentrated in fast-interconnect markets (ERCOT, Phoenix) where
    # hyperscalers are actively building, not in PJM-constrained markets
    # like NOVA where the queue backlog limits new project starts.
    #
    # Sources: JLL North America DC Report (Jan 2026), CBRE DC Market
    # Tracker (H2 2025), PJM/ERCOT interconnection queue statistics.
    preseed_completions = compute_preseed_completions(t_end=t_end)
    preseed_shares = {
        # US regions: pipeline-adjusted shares reflecting the shift toward
        # fast-interconnect markets while acknowledging NOVA's large existing
        # pipeline of projects already past PJM queue and under construction.
        #
        # Shifted from pure capacity-proportional (which over-weights NOVA
        # at 32%) toward pipeline geography (DFW/ERCOT/PHX growing share).
        # NOVA reduced from 32% to 27%; ERCOT markets increased.
        #
        # Sensitivity: with capacity-proportional preseed (60% moratorium),
        # ADT delay is ~5.7wk. With this distribution (52% moratorium),
        # ADT delay is ~2.1wk. The difference is driven by how much
        # "counterfactual capacity" exists in moratorium-affected regions.
        #
        # Source: JLL North America DC Report (Jan 2026), CBRE DC Market
        # Tracker (H2 2025), PJM/ERCOT interconnection queue statistics.
        "NOVA":       0.27,   # Largest base; many projects under construction
        "NYC_NJ":     0.06,   # Modest pipeline; expensive market
        "CHI":        0.05,   # Moderate; 350 E Cermak area expansions
        "WEST_COAST": 0.09,   # WA (Quincy) + OR active; CA constrained
        "OTHER_BLUE": 0.05,   # Scattered small projects
        "DFW":        0.17,   # Largest growth market; ERCOT fast-track
        "PHX":        0.09,   # Growing market; Mesa/Goodyear campuses
        "ATL":        0.06,   # Growing Southeast hub
        "TX_OTHER":   0.07,   # Houston, W. TX renewable-powered sites
        "OTHER_RED":  0.09,   # UT, NV, ID, SC, TN growth markets
        # International: no preseed (not modeled)
        "PERSIAN_GULF": 0.0,
        "CANADA":     0.0,
        "MEXICO":     0.0,
    }

    results = SimulationResults(scenario_name=scenario.name, scenario=scenario, t_end=t_end)

    capacity_ts = np.zeros(t_end)
    us_capacity_ts = np.zeros(t_end)
    qw_capacity_ts = np.zeros(t_end)
    blocked_ts = np.zeros(t_end)
    capacity_by_region = {r: np.zeros(t_end) for r in REGIONS}

    for t in range(t_end):
        # Investment elasticity: policy uncertainty dampens total sector capex
        baseline_shares = displacement.allocate_baseline(t)
        moratorium_coverage = sum(
            baseline_shares[r] * displacement.get_moratorium_strength(r, t, scenario)
            for r in REGIONS if not REGIONS[r].is_international
        )
        dampened_investment = investment_curve[t] * (1 - inv_elasticity * moratorium_coverage)

        # Module A: Allocate investment
        a_result = displacement.allocate_with_moratoria(t, scenario, dampened_investment)
        for region, mw in a_result["allocation"].items():
            capacity_model.add_to_pipeline(region, mw, t)

        # Pre-seeded completions
        preseed = {r: preseed_completions[t] * preseed_shares[r] for r in REGIONS}

        # Module B: Process completions and depreciation
        b_result = capacity_model.step(t, preseed_completions=preseed)

        # Record trajectories
        capacity_ts[t] = b_result["total_capacity"]
        us_capacity_ts[t] = sum(
            b_result["capacity"][r] for r in REGIONS if not REGIONS[r].is_international
        )
        qw_capacity_ts[t] = sum(
            b_result["capacity"][r] * REGIONS[r].compute_quality for r in REGIONS
        )
        blocked_ts[t] = sum(a_result["blocked"].values())
        for region in REGIONS:
            capacity_by_region[region][t] = b_result["capacity"][region]

        # Update agglomeration
        moratoria_strength = {
            r: displacement.get_moratorium_strength(r, t, scenario) for r in REGIONS
        }
        displacement.update_state(b_result["capacity"], b_result["capacity_changes"], moratoria_strength)

    results.capacity_trajectory = capacity_ts
    results.us_capacity_trajectory = us_capacity_ts
    results.quality_weighted_trajectory = qw_capacity_ts
    results.capacity_by_region = capacity_by_region
    results.blocked_trajectory = blocked_ts

    # Module C: Hardware compute timeline
    results.ai_timeline = run_compute(qw_capacity_ts, scenario_name=scenario.name)

    return results


def run_all_scenarios(
    scenarios: dict[str, Scenario] = None,
    t_end: int = T_END,
    **kwargs,
) -> dict[str, SimulationResults]:
    """Run all scenarios and compute delays relative to baseline."""
    scens = scenarios or SCENARIOS
    all_results = {}

    baseline_results = run_scenario(scens["baseline"], t_end=t_end, **kwargs)
    all_results["baseline"] = baseline_results

    for name, scenario in scens.items():
        if name == "baseline":
            continue
        result = run_scenario(scenario, t_end=t_end, **kwargs)
        compare_to_baseline(baseline_results.ai_timeline, result.ai_timeline)
        all_results[name] = result

    return all_results
