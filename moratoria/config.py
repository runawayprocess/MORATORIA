"""
Central configuration for the Data Center Moratoria Impact Model.

All parameters are literature-derived defaults. No formal calibration step.
The model's value is in scenario comparison (moratorium vs. baseline), not point forecasts.

Sources:
  - Epoch AI GATE model (2025): hardware/software efficiency, compute depreciation
  - JLL/CBRE DC reports (2025): regional capacity, construction timelines
  - PJM/ERCOT public data: interconnection queue times
  - Greenstone 2002, Hsieh-Moretti 2019: displacement/misallocation parameters
  - DOE/S&P Global/Gartner: 2030 capacity projections
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Time horizon
# ---------------------------------------------------------------------------
T_START = 0       # Q1 2026
T_END = 28        # Q4 2032 (28 quarters)
QUARTERS_PER_YEAR = 4
BASE_YEAR = 2026
BASE_QUARTER = 1  # Q1


def quarter_label(t: int) -> str:
    """Convert simulation quarter index to human-readable label."""
    year = BASE_YEAR + (BASE_QUARTER - 1 + t) // QUARTERS_PER_YEAR
    q = (BASE_QUARTER - 1 + t) % QUARTERS_PER_YEAR + 1
    return f"Q{q} {year}"


# ---------------------------------------------------------------------------
# Module A: Geographic Displacement Parameters
# ---------------------------------------------------------------------------

@dataclass
class RegionParams:
    """Parameters for a single data center market region."""
    name: str
    short_name: str
    political_lean: str               # "blue", "red", "swing", "international"
    capacity_2025_mw: float           # Installed capacity end of 2025 (MW)
    grid_speed_score: float           # 0-1, higher = faster interconnection
    fiber_density_score: float        # 0-1, higher = denser fiber/IX ecosystem
    cost_score: float                 # 0-1, higher = cheaper (land+power+tax)
    agglomeration_score_init: float   # 0-1, initial agglomeration strength
    policy_score: float               # 0-1, higher = more favorable policy
    water_score: float                # 0-1, higher = better water availability
    labor_score: float                # 0-1, higher = better labor market
    interconnect_time_qtrs: float     # Base interconnection queue time (quarters)
    build_time_qtrs: float            # Construction time after interconnection (quarters)
    iso_rto: str                      # Grid operator
    compute_quality: float = 1.0      # 0-1, effective compute per MW (fiber/IX proximity)
    is_international: bool = False    # True for non-US regions


# Literature-derived logit weights (relative importance of each factor).
LOGIT_WEIGHTS = {
    "grid": 0.25,          # Grid speed is dominant constraint (RMI PJM analysis)
    "fiber": 0.15,         # Fiber ecosystem matters for cloud/enterprise
    "cost": 0.15,          # Land + power + tax competitiveness
    "agglomeration": 0.20, # Krugman NEG: co-location benefits are strong
    "policy": 0.10,        # Tax incentives, regulatory climate
    "water": 0.08,         # Growing constraint but not yet dominant
    "labor": 0.07,         # Skilled labor availability
}

LOGIT_TEMPERATURE = 0.15

# Agglomeration dynamics (calibrated from DFW trajectory, 2018-2025)
AGGLOMERATION_BUILD_RATE = 0.06
AGGLOMERATION_DECAY_RATE = 0.04
AGGLOMERATION_ELASTICITY = 0.4

# Reallocation friction
REALLOCATION_DELAY_QTRS = 3

# Workload fungibility: weighted-average fraction that can relocate (~0.556)
WORKLOAD_MIX = {
    "ai_training":     {"share": 0.40, "fungibility": 0.85},
    "cloud_compute":   {"share": 0.30, "fungibility": 0.60},
    "enterprise_colo": {"share": 0.15, "fungibility": 0.30},
    "edge_lowlatency": {"share": 0.10, "fungibility": 0.10},
    "government":      {"share": 0.05, "fungibility": 0.02},
}
EFFECTIVE_FUNGIBILITY = sum(
    v["share"] * v["fungibility"] for v in WORKLOAD_MIX.values()
)


# ---------------------------------------------------------------------------
# Module B: Aggregate Capacity Parameters
# ---------------------------------------------------------------------------

CAPACITY_2025_US_MW = 25_000
CAPACITY_2030_TARGET_MW = 55_000
DEPRECIATION_RATE_QTR = 0.01        # ~4% annual (25-year facility lifecycle)
BASELINE_SLIPPAGE_RATE = 0.35

# International organic growth (Gulf/Canada/Mexico grow independently of US investment)
INTL_ORGANIC_GROWTH_QTR = 0.04      # ~17% annual


# ---------------------------------------------------------------------------
# Module C: Hardware Compute Parameters
# ---------------------------------------------------------------------------
# Algorithmic/software efficiency is NOT modeled in the simulation.
# Hardware compute only. Software is reported as analytical context.

# AI's share of total DC capacity (linear 20% -> 40% over 2025-2030)
AI_SHARE_2025 = 0.20
AI_SHARE_2030 = 0.40

# Effective FLOP/s per MW of AI-dedicated capacity (2025 baseline)
# Derived from: (1/PUE) * GPU_density * GPU_perf
# = (1/1.20) * 400 GPUs/MW * 1e15 FLOP/s/GPU = 3.33e17
FLOPS_PER_AI_MW_2025 = (1 / 1.20) * 400 * 1e15

# Compound quarterly improvement in FLOP/s per MW
# PUE improvement (~0.4%/qtr) + GPU density (~2%/qtr) + GPU performance (~6.8%/qtr)
HARDWARE_IMPROVEMENT_QTR = 0.094

# Training run parameters
GPU_UTILIZATION_TRAINING = 0.40
TRAINING_DURATION_MONTHS = 4
TRAINING_DURATION_SECONDS = TRAINING_DURATION_MONTHS * 30.44 * 24 * 3600
FRONTIER_TRAINING_FRACTION = 0.08

# Hardware-only milestones (multiples of starting hardware training FLOP)
HARDWARE_MILESTONES = {
    "2x_hardware":  1.2e27,
    "5x_hardware":  3e27,
    "10x_hardware": 6e27,
    "20x_hardware": 1.2e28,
}

# Hardware efficiency growth (used in analytical decomposition only)
HARDWARE_EFFICIENCY_GROWTH_ANNUAL = 0.30

# Software parameters (analytical decomposition ONLY, not simulated)
ALGO_DOUBLING_TIME_MONTHS = 12
