"""
Compute model: capacity trajectory -> AI capabilities timeline.

Maps quality-weighted DC capacity (MW) to hardware FLOP/s, applies algorithmic
efficiency multiplier, and finds milestone crossing times for effective compute.

Key insight: moratoriums affect hardware deployment but not algorithmic research.
Algorithmic efficiency doubles every ~12 months (Erdil & Besiroglu 2024) and
is treated as exogenous to the simulation. This is a simplification: some
algorithmic progress depends on hardware compute availability for experiments.
We tested an endogeneity parameter (ALGO_ENDOGENEITY = 0.15) but the effect
was negligible (< 2% difference in algo multiplier between baseline and the
broadest moratorium scenario). The parameter was removed for transparency.

Sources:
- Epoch AI (2025): GATE model, hardware efficiency trends
- Epoch AI (2025): "Can AI Scaling Continue Through 2030?"
- Erdil & Besiroglu (2024): algorithmic progress in language models
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from moratoria.config import (
    AI_SHARE_2025,
    AI_SHARE_2030,
    FLOPS_PER_AI_MW_2025,
    HARDWARE_IMPROVEMENT_QTR,
    GPU_UTILIZATION_TRAINING,
    TRAINING_DURATION_SECONDS,
    FRONTIER_TRAINING_FRACTION,
    HARDWARE_MILESTONES,
    EFFECTIVE_COMPUTE_MULTIPLIERS,
    HARDWARE_EFFICIENCY_GROWTH_ANNUAL,
    ALGO_DOUBLING_TIME_MONTHS,
    quarter_label,
)


# ---------------------------------------------------------------------------
# Core compute functions
# ---------------------------------------------------------------------------

def compute_effective_flops(capacity_mw: float, t: int) -> float:
    """
    Effective hardware FLOP/s available for AI workloads at quarter t.

    Q(t) = K_total * alpha_AI(t) * flops_per_mw(t)

    alpha_AI grows linearly from 20% to 40% over 20 quarters (2025-2030).
    flops_per_mw grows at ~9.4%/quarter (compound of PUE + density + GPU perf).
    """
    t_saturate = 20
    alpha = min(AI_SHARE_2030, AI_SHARE_2025 + (AI_SHARE_2030 - AI_SHARE_2025) * t / t_saturate)
    flops_per_mw = FLOPS_PER_AI_MW_2025 * (1 + HARDWARE_IMPROVEMENT_QTR) ** t
    return capacity_mw * alpha * flops_per_mw


def compute_training_flops(effective_flops: float) -> float:
    """Total hardware FLOP for a single frontier training run."""
    return effective_flops * FRONTIER_TRAINING_FRACTION * GPU_UTILIZATION_TRAINING * TRAINING_DURATION_SECONDS


# ---------------------------------------------------------------------------
# Algorithmic efficiency (exogenous)
# ---------------------------------------------------------------------------

def compute_algo_multiplier(t_end: int) -> np.ndarray:
    """
    Compute algorithmic efficiency multiplier over time.

    Purely exogenous: doubles every ALGO_DOUBLING_TIME_MONTHS months
    (~19%/quarter). Identical across all scenarios. Moratoriums do not
    affect algorithmic progress in this model.
    """
    growth_qtr = 2 ** (3 / ALGO_DOUBLING_TIME_MONTHS) - 1
    return np.array([(1 + growth_qtr) ** t for t in range(t_end)])


# ---------------------------------------------------------------------------
# Trajectory and milestone computation
# ---------------------------------------------------------------------------

def compute_trajectories(capacity_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute FLOP/s and training FLOP trajectories from capacity trajectory."""
    t_end = len(capacity_trajectory)
    flops = np.array([compute_effective_flops(capacity_trajectory[t], t) for t in range(t_end)])
    training = np.array([compute_training_flops(f) for f in flops])
    return flops, training


def _interpolate_crossing(trajectory: np.ndarray, threshold: float) -> Optional[float]:
    """Find interpolated time at which trajectory crosses threshold (log-linear)."""
    for i in range(len(trajectory)):
        if trajectory[i] >= threshold:
            if i == 0 or trajectory[i - 1] >= threshold:
                return float(i)
            log_prev = np.log(max(trajectory[i - 1], 1e-300))
            log_curr = np.log(trajectory[i])
            log_thresh = np.log(threshold)
            if log_curr == log_prev:
                return float(i)
            frac = (log_thresh - log_prev) / (log_curr - log_prev)
            return (i - 1) + frac
    return None


def find_milestones(trajectory: np.ndarray, milestones: dict = None) -> dict[str, Optional[int]]:
    """Find the integer quarter at which each milestone threshold is reached."""
    ms = milestones or HARDWARE_MILESTONES
    results = {}
    for name, threshold in ms.items():
        reached = np.where(trajectory >= threshold)[0]
        results[name] = int(reached[0]) if len(reached) > 0 else None
    return results


def compute_milestone_delays(
    baseline_milestones: dict,
    scenario_milestones: dict,
    baseline_trajectory: np.ndarray,
    scenario_trajectory: np.ndarray,
    milestones_dict: dict = None,
) -> dict[str, dict]:
    """Compute interpolated delays by comparing scenario to baseline."""
    ms = milestones_dict or HARDWARE_MILESTONES
    results = {}
    for name in baseline_milestones:
        bt = baseline_milestones[name]
        st = scenario_milestones.get(name)

        if bt is None or st is None:
            results[name] = {
                "baseline_t": bt, "scenario_t": st,
                "delay_qtrs_interpolated": None, "delay_months_interpolated": None,
                "delay_weeks_interpolated": None,
                "shortfall_pct_at_baseline_crossing": None,
            }
            continue

        bi = _interpolate_crossing(baseline_trajectory, ms[name])
        si = _interpolate_crossing(scenario_trajectory, ms[name])
        delay_interp = (si - bi) if (bi is not None and si is not None) else None

        shortfall_pct = None
        if bt < len(baseline_trajectory):
            bl_val = baseline_trajectory[bt]
            sc_val = scenario_trajectory[bt]
            if bl_val > 0:
                shortfall_pct = max(0, (1 - sc_val / bl_val) * 100)

        d = max(0, delay_interp) if delay_interp is not None else None
        results[name] = {
            "baseline_t": bt, "scenario_t": st,
            "delay_qtrs_interpolated": round(d, 3) if d is not None else None,
            "delay_months_interpolated": round(d * 3, 2) if d is not None else None,
            "delay_weeks_interpolated": round(d * 13, 1) if d is not None else None,
            "shortfall_pct_at_baseline_crossing": round(shortfall_pct, 1) if shortfall_pct is not None else None,
        }
    return results


# ---------------------------------------------------------------------------
# Results dataclass and runner
# ---------------------------------------------------------------------------

@dataclass
class AITimelineResults:
    """Complete compute results for one scenario."""
    scenario_name: str
    capacity_trajectory_mw: np.ndarray
    effective_flops_trajectory: np.ndarray
    training_flops_trajectory: np.ndarray
    milestone_quarters: dict[str, Optional[int]]

    # Effective compute (hardware * algo)
    algo_multiplier: Optional[np.ndarray] = None
    effective_compute_trajectory: Optional[np.ndarray] = None
    effective_milestones: Optional[dict] = None
    effective_milestone_delays: Optional[dict] = None

    # Delays and shortfalls (vs baseline)
    milestone_delays: Optional[dict] = None
    capacity_shortfall_pct: Optional[np.ndarray] = None
    compute_shortfall_pct: Optional[np.ndarray] = None
    peak_delay_qtrs: Optional[float] = None  # max time-delay at any point
    delay_trajectory: Optional[np.ndarray] = None  # delay at each quarter

    # Cumulative metrics (what the back-of-envelope can't produce)
    cumulative_flop_deficit: Optional[float] = None       # integral of (baseline - scenario) eff compute
    cumulative_flop_deficit_pct: Optional[float] = None   # as % of cumulative baseline
    reequilibration_quarter: Optional[int] = None         # first quarter where shortfall < 1%
    peak_shortfall_quarter: Optional[int] = None          # quarter of peak capacity shortfall

    def summary(self) -> str:
        lines = [f"=== Compute Results: {self.scenario_name} ===\n"]
        lines.append(
            f"Quality-weighted capacity: {self.capacity_trajectory_mw[0]/1000:.1f} GW -> "
            f"{self.capacity_trajectory_mw[-1]/1000:.1f} GW"
        )
        lines.append(
            f"Hardware training FLOP: {self.training_flops_trajectory[0]:.2e} -> "
            f"{self.training_flops_trajectory[-1]:.2e}"
        )

        if self.effective_compute_trajectory is not None:
            lines.append(
                f"Effective training FLOP (hw*algo): {self.effective_compute_trajectory[0]:.2e} -> "
                f"{self.effective_compute_trajectory[-1]:.2e}"
            )
            if self.algo_multiplier is not None:
                lines.append(f"Algo efficiency multiplier: 1.0x -> {self.algo_multiplier[-1]:.1f}x")

        if self.peak_delay_qtrs is not None:
            days = self.peak_delay_qtrs * 91.25
            weeks = days / 7
            lines.append(f"\nPeak delay vs. baseline: {self.peak_delay_qtrs:.2f} quarters (~{weeks:.1f} weeks / {days:.0f} days)")
            if self.peak_shortfall_quarter is not None:
                lines.append(f"Peak capacity shortfall at: {quarter_label(self.peak_shortfall_quarter)}")

        if self.cumulative_flop_deficit_pct is not None:
            lines.append(f"Cumulative effective compute deficit: {self.cumulative_flop_deficit_pct:.1f}% of baseline total")

        if self.reequilibration_quarter is not None:
            lines.append(f"Compute shortfall drops below 1% at: {quarter_label(self.reequilibration_quarter)}")
        elif self.compute_shortfall_pct is not None and np.any(self.compute_shortfall_pct > 1.0):
            lines.append(f"Compute shortfall does NOT drop below 1% within simulation window")

        if self.effective_milestone_delays:
            lines.append("\nEffective Compute Delays vs. Baseline:")
            for name, info in self.effective_milestone_delays.items():
                d = info.get("delay_qtrs_interpolated")
                w = info.get("delay_weeks_interpolated")
                if d is not None:
                    lines.append(f"  {name}: {d:.3f} quarters (~{w:.1f} weeks)")
                else:
                    lines.append(f"  {name}: N/A")
        elif self.milestone_delays:
            lines.append("\nHardware Delays vs. Baseline:")
            for name, info in self.milestone_delays.items():
                d = info.get("delay_qtrs_interpolated")
                if d is not None:
                    lines.append(f"  {name}: {d:.2f} quarters ({d*3:.1f} months)")
                else:
                    lines.append(f"  {name}: N/A")

        lines.append("\nHardware Milestones:")
        for name, t in self.milestone_quarters.items():
            if t is not None:
                lines.append(f"  {name}: reached at {quarter_label(t)}")
            else:
                lines.append(f"  {name}: NOT reached")

        return "\n".join(lines)


def run_compute(capacity_trajectory_mw: np.ndarray, scenario_name: str = "unnamed") -> AITimelineResults:
    """Map quality-weighted capacity to hardware compute timeline."""
    flops, training = compute_trajectories(capacity_trajectory_mw)
    milestones = find_milestones(training)
    return AITimelineResults(
        scenario_name=scenario_name,
        capacity_trajectory_mw=capacity_trajectory_mw,
        effective_flops_trajectory=flops,
        training_flops_trajectory=training,
        milestone_quarters=milestones,
    )


def compare_to_baseline(baseline: AITimelineResults, scenario: AITimelineResults):
    """
    Compute delays and shortfalls by comparing scenario to baseline.

    Computes effective compute (hardware * algo) for both scenarios,
    where algo progress is partially endogenous to hardware availability.
    """
    # Hardware milestone delays (diagnostic)
    scenario.milestone_delays = compute_milestone_delays(
        baseline.milestone_quarters, scenario.milestone_quarters,
        baseline.training_flops_trajectory, scenario.training_flops_trajectory,
    )

    # Algorithmic efficiency: exogenous, identical across all scenarios.
    t_end = len(baseline.training_flops_trajectory)
    algo = compute_algo_multiplier(t_end)

    # Baseline effective compute (computed once)
    if baseline.effective_compute_trajectory is None:
        baseline.algo_multiplier = algo
        baseline.effective_compute_trajectory = baseline.training_flops_trajectory * algo

        starting_effective = baseline.effective_compute_trajectory[0]
        eff_milestones = {n: starting_effective * m for n, m in EFFECTIVE_COMPUTE_MULTIPLIERS.items()}
        baseline.effective_milestones = find_milestones(baseline.effective_compute_trajectory, eff_milestones)

    # Scenario effective compute (same algo, different hardware)
    scenario.algo_multiplier = algo
    scenario.effective_compute_trajectory = scenario.training_flops_trajectory * algo

    # Effective compute milestones
    starting_effective = baseline.effective_compute_trajectory[0]
    eff_milestones = {n: starting_effective * m for n, m in EFFECTIVE_COMPUTE_MULTIPLIERS.items()}
    scenario_eff_ms = find_milestones(scenario.effective_compute_trajectory, eff_milestones)
    scenario.effective_milestones = scenario_eff_ms

    scenario.effective_milestone_delays = compute_milestone_delays(
        baseline.effective_milestones, scenario_eff_ms,
        baseline.effective_compute_trajectory, scenario.effective_compute_trajectory,
        milestones_dict=eff_milestones,
    )

    # Peak delay: at each quarter, how many quarters is the scenario "behind"?
    # delay(t) ≈ log(baseline/scenario) / quarterly_log_growth_rate
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log(np.maximum(baseline.effective_compute_trajectory, 1e-300)) - \
                    np.log(np.maximum(scenario.effective_compute_trajectory, 1e-300))
        # Compute average quarterly log growth rate from baseline
        log_growth = np.diff(np.log(np.maximum(baseline.effective_compute_trajectory, 1e-300)))
        # Use rolling growth rate, avoid division by zero
        delay_trajectory = np.zeros(len(log_ratio))
        for t in range(1, len(log_ratio)):
            g = log_growth[t - 1] if t - 1 < len(log_growth) and log_growth[t - 1] > 0 else 0.3
            delay_trajectory[t] = max(0, log_ratio[t] / g)
        scenario.delay_trajectory = delay_trajectory
        scenario.peak_delay_qtrs = round(float(np.max(delay_trajectory)), 3)

    # Shortfalls
    with np.errstate(divide='ignore', invalid='ignore'):
        scenario.capacity_shortfall_pct = np.where(
            baseline.capacity_trajectory_mw > 0,
            (1 - scenario.capacity_trajectory_mw / baseline.capacity_trajectory_mw) * 100, 0)
        scenario.compute_shortfall_pct = np.where(
            baseline.effective_compute_trajectory > 0,
            (1 - scenario.effective_compute_trajectory / baseline.effective_compute_trajectory) * 100, 0)

    # Peak shortfall timing
    if scenario.capacity_shortfall_pct is not None:
        scenario.peak_shortfall_quarter = int(np.argmax(scenario.capacity_shortfall_pct))

    # Cumulative effective compute deficit: integral of (baseline - scenario)
    # This captures the total "lost compute" over the planning horizon, not just
    # the instantaneous shortfall. A 5% shortfall sustained for 20 quarters is
    # worse than a 10% shortfall for 2 quarters.
    deficit = np.maximum(0, baseline.effective_compute_trajectory - scenario.effective_compute_trajectory)
    scenario.cumulative_flop_deficit = float(np.sum(deficit))
    cumulative_baseline = float(np.sum(baseline.effective_compute_trajectory))
    if cumulative_baseline > 0:
        scenario.cumulative_flop_deficit_pct = round(
            scenario.cumulative_flop_deficit / cumulative_baseline * 100, 1)

    # Reequilibration: first quarter where compute shortfall drops below 1%
    # after having been above 1%. This shows whether the impact is transient.
    above_threshold = scenario.compute_shortfall_pct > 1.0
    if np.any(above_threshold):
        last_above = int(np.where(above_threshold)[0][-1])
        if last_above < len(scenario.compute_shortfall_pct) - 1:
            scenario.reequilibration_quarter = last_above + 1
        # else: shortfall still > 1% at end of simulation


# ---------------------------------------------------------------------------
# Analytical decomposition (for the report)
# ---------------------------------------------------------------------------

def decompose_compute_growth(
    algo_doubling_months: float = ALGO_DOUBLING_TIME_MONTHS,
    hardware_growth_annual: float = HARDWARE_EFFICIENCY_GROWTH_ANNUAL,
) -> dict:
    """
    Decompose effective compute growth into hardware vs. software shares.
    """
    hw_annual = hardware_growth_annual
    sw_annual = 2 ** (12 / algo_doubling_months) - 1

    total = (1 + hw_annual) * (1 + sw_annual) - 1
    log_total = np.log(1 + total)
    hw_share = np.log(1 + hw_annual) / log_total
    sw_share = np.log(1 + sw_annual) / log_total

    return {
        "hardware_growth_annual": hw_annual,
        "software_growth_annual": sw_annual,
        "total_growth_annual": total,
        "hardware_share": hw_share,
        "software_share": sw_share,
        "effective_delay_fraction": hw_share,
    }
