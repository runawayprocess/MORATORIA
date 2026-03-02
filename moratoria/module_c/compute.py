"""
Hardware compute model: capacity trajectory -> AI compute timeline.

Maps quality-weighted DC capacity (MW) to effective hardware FLOP/s, then to
frontier training run FLOP, and finds milestone crossing times.

Software/algorithmic efficiency is NOT modeled; it is reported separately
via decompose_compute_growth() as analytical context.

Sources:
- Epoch AI (2025): GATE model, hardware efficiency trends
- Epoch AI (2025): "Can AI Scaling Continue Through 2030?"
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
    t_saturate = 20  # AI share reaches 40% at Q4 2029
    alpha = min(AI_SHARE_2030, AI_SHARE_2025 + (AI_SHARE_2030 - AI_SHARE_2025) * t / t_saturate)
    flops_per_mw = FLOPS_PER_AI_MW_2025 * (1 + HARDWARE_IMPROVEMENT_QTR) ** t
    return capacity_mw * alpha * flops_per_mw


def compute_training_flops(effective_flops: float) -> float:
    """Total hardware FLOP for a single frontier training run."""
    return effective_flops * FRONTIER_TRAINING_FRACTION * GPU_UTILIZATION_TRAINING * TRAINING_DURATION_SECONDS


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


def find_milestones(training_trajectory: np.ndarray) -> dict[str, Optional[int]]:
    """Find the integer quarter at which each FLOP milestone is reached."""
    results = {}
    for name, threshold in HARDWARE_MILESTONES.items():
        reached = np.where(training_trajectory >= threshold)[0]
        results[name] = int(reached[0]) if len(reached) > 0 else None
    return results


def compute_milestone_delays(
    baseline_milestones: dict,
    scenario_milestones: dict,
    baseline_trajectory: np.ndarray,
    scenario_trajectory: np.ndarray,
) -> dict[str, dict]:
    """Compute delays and shortfalls by comparing scenario to baseline."""
    results = {}
    for name in baseline_milestones:
        bt = baseline_milestones[name]
        st = scenario_milestones.get(name)

        if bt is None or st is None:
            results[name] = {
                "baseline_t": bt, "scenario_t": st,
                "delay_qtrs_interpolated": None, "delay_months_interpolated": None,
                "shortfall_pct_at_baseline_crossing": None,
            }
            continue

        # Interpolated delay
        bi = _interpolate_crossing(baseline_trajectory, HARDWARE_MILESTONES[name])
        si = _interpolate_crossing(scenario_trajectory, HARDWARE_MILESTONES[name])
        delay_interp = (si - bi) if (bi is not None and si is not None) else None
        if delay_interp is not None and abs(delay_interp) < 0.05:
            delay_interp = 0.0

        # Shortfall at baseline crossing
        shortfall_pct = None
        if bt < len(baseline_trajectory):
            bl_val = baseline_trajectory[bt]
            sc_val = scenario_trajectory[bt]
            if bl_val > 0:
                shortfall_pct = max(0, (1 - sc_val / bl_val) * 100)

        results[name] = {
            "baseline_t": bt, "scenario_t": st,
            "delay_qtrs_interpolated": round(max(0, delay_interp), 2) if delay_interp is not None else None,
            "delay_months_interpolated": round(max(0, delay_interp * 3), 1) if delay_interp is not None else None,
            "shortfall_pct_at_baseline_crossing": round(shortfall_pct, 1) if shortfall_pct is not None else None,
        }
    return results


# ---------------------------------------------------------------------------
# Results dataclass and runner
# ---------------------------------------------------------------------------

@dataclass
class AITimelineResults:
    """Complete hardware compute results for one scenario."""
    scenario_name: str
    capacity_trajectory_mw: np.ndarray
    effective_flops_trajectory: np.ndarray
    training_flops_trajectory: np.ndarray
    milestone_quarters: dict[str, Optional[int]]
    milestone_delays: Optional[dict] = None
    capacity_shortfall_pct: Optional[np.ndarray] = None
    compute_shortfall_pct: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [f"=== Hardware Compute Results: {self.scenario_name} ===\n"]
        lines.append(
            f"Quality-weighted capacity: {self.capacity_trajectory_mw[0]/1000:.1f} GW -> "
            f"{self.capacity_trajectory_mw[-1]/1000:.1f} GW"
        )
        lines.append(
            f"AI hardware FLOP/s: {self.effective_flops_trajectory[0]:.2e} -> "
            f"{self.effective_flops_trajectory[-1]:.2e}"
        )
        lines.append(
            f"Hardware training FLOP: {self.training_flops_trajectory[0]:.2e} -> "
            f"{self.training_flops_trajectory[-1]:.2e}"
        )
        lines.append("\nHardware Milestones:")
        for name, t in self.milestone_quarters.items():
            if t is not None:
                lines.append(f"  {name}: reached at {quarter_label(t)}")
            else:
                lines.append(f"  {name}: NOT reached within simulation window")

        if self.milestone_delays:
            lines.append("\nHardware Delays vs. Baseline:")
            for name, info in self.milestone_delays.items():
                d = info.get("delay_qtrs_interpolated")
                shortfall = info.get("shortfall_pct_at_baseline_crossing")
                if d is not None:
                    s_str = f", compute shortfall {shortfall:.1f}%" if shortfall else ""
                    lines.append(f"  {name}: {d:.1f} quarters ({d*3:.0f} months){s_str}")
                else:
                    lines.append(f"  {name}: N/A")
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
    """Compute delays and shortfalls by comparing scenario to baseline."""
    scenario.milestone_delays = compute_milestone_delays(
        baseline.milestone_quarters, scenario.milestone_quarters,
        baseline.training_flops_trajectory, scenario.training_flops_trajectory,
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        scenario.capacity_shortfall_pct = np.where(
            baseline.capacity_trajectory_mw > 0,
            (1 - scenario.capacity_trajectory_mw / baseline.capacity_trajectory_mw) * 100, 0)
        scenario.compute_shortfall_pct = np.where(
            baseline.training_flops_trajectory > 0,
            (1 - scenario.training_flops_trajectory / baseline.training_flops_trajectory) * 100, 0)


# ---------------------------------------------------------------------------
# Analytical decomposition (for the report, not simulated)
# ---------------------------------------------------------------------------

def decompose_compute_growth(
    algo_doubling_months: float = ALGO_DOUBLING_TIME_MONTHS,
    hardware_growth_annual: float = HARDWARE_EFFICIENCY_GROWTH_ANNUAL,
) -> dict:
    """
    Decompose effective compute growth into hardware vs. software shares.
    This is an ANALYTICAL tool for the report, not part of the simulation.
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
