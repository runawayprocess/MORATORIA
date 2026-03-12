"""
Aggregate capacity model with pipeline tracking.

Tracks MW in construction pipeline by region, handles completions with
region-specific construction timelines, and applies depreciation.

Two congestion mechanisms:
1. Queue congestion: interconnection queue times increase when regional
   pipeline-to-capacity ratio rises (PJM queues, permitting backlogs).
2. Labor/equipment congestion: build times increase when a region's share
   of national construction activity exceeds its historical share, reflecting
   construction workforce scarcity from concentrated demand.

Theoretical grounding:
- Hsieh & Moretti (2019): localized supply constraints cause aggregate
  misallocation even when activity can in principle relocate
- Turner & Townsend (2025): DC construction costs rise 15-20% when
  regional pipeline exceeds historical norms
"""

from collections import defaultdict

from moratoria.config import DEPRECIATION_RATE_QTR, T_END
from moratoria.data.regions import REGIONS, RegionParams

# Queue congestion: 30% pipeline-to-capacity increase -> 30% longer interconnection
CONGESTION_SENSITIVITY = 0.3

# Queue congestion threshold: the pipeline-to-capacity ratio at which
# interconnection delays begin extending. Below this ratio, the interconnection
# system can absorb new projects at the base rate. Above it, queue congestion
# kicks in proportionally.
#
# Range: 0.08-0.30 (swept in Sobol analysis)
#   0.08 = very tight: even modest pipeline causes congestion (aggressive)
#   0.15 = moderate: congestion fires when pipeline exceeds ~15% of capacity (default)
#   0.30 = loose: region can absorb large pipeline before congestion appears
#
# At 0.15, DFW (3,500 MW) hits congestion when pipeline exceeds 525 MW.
# At 0.30, threshold rises to 1,050 MW. The parameter interacts strongly
# with CONGESTION_SENSITIVITY.
CONGESTION_THRESHOLD = 0.15

# Labor/equipment congestion: when a region's share of national construction
# pipeline exceeds its historical capacity share, build times extend.
# 0.15 = 15% longer build time per 100% excess concentration.
# Based on Turner & Townsend DC Cost Index showing 15-20% cost increases
# translating to ~15% timeline extension under concentrated demand.
LABOR_CONGESTION_SENSITIVITY = 0.15


class CapacityModel:
    """
    Pipeline-based capacity accumulation model.

    Tracks investment entering the construction pipeline and completing
    after region-specific delays (interconnect queue + build time).
    """

    def __init__(self, regions: dict[str, RegionParams] = None):
        self.regions = regions or REGIONS
        self.region_names = list(self.regions.keys())

        self.capacity: dict[str, float] = {
            name: r.capacity_2025_mw for name, r in self.regions.items()
        }
        self.pipeline: dict[tuple[str, int], float] = defaultdict(float)
        self.pipeline_by_region: dict[str, float] = defaultdict(float)

        # Historical capacity shares for labor congestion baseline
        total_cap = sum(r.capacity_2025_mw for r in self.regions.values())
        self.historical_shares = {
            name: r.capacity_2025_mw / total_cap
            for name, r in self.regions.items()
        }

    def add_to_pipeline(self, region: str, mw: float, t: int):
        """Add MW to a region's construction pipeline at time t."""
        if mw <= 0:
            return
        r = self.regions[region]

        # Queue congestion: interconnection times increase with pipeline backlog
        baseline_pipeline = self.capacity[region] * CONGESTION_THRESHOLD
        ratio = self.pipeline_by_region[region] / max(baseline_pipeline, 1)
        queue_congestion = CONGESTION_SENSITIVITY * max(0, ratio - 1)
        interconnect_time = r.interconnect_time_qtrs * (1 + queue_congestion)

        # Labor/equipment congestion: build times increase when a region
        # absorbs a disproportionate share of national construction activity.
        # This captures the construction workforce bottleneck when moratoriums
        # concentrate demand in fewer regions.
        national_pipeline = sum(self.pipeline_by_region.values())
        if national_pipeline > 0:
            actual_share = self.pipeline_by_region[region] / national_pipeline
            historical = max(self.historical_shares.get(region, 0.05), 0.02)
            excess_concentration = max(0, actual_share / historical - 1.0)
            labor_congestion = LABOR_CONGESTION_SENSITIVITY * excess_concentration
        else:
            labor_congestion = 0
        build_time = r.build_time_qtrs * (1 + labor_congestion)

        construction_time = interconnect_time + build_time

        t_floor = t + int(construction_time)
        t_ceil = t_floor + 1
        frac_ceil = construction_time - int(construction_time)

        if t_floor <= T_END + 8:
            self.pipeline[(region, t_floor)] += mw * (1.0 - frac_ceil)
        if t_ceil <= T_END + 8 and frac_ceil > 0:
            self.pipeline[(region, t_ceil)] += mw * frac_ceil
        self.pipeline_by_region[region] += mw

    def step(self, t: int, preseed_completions: dict[str, float] = None) -> dict:
        """Advance one quarter: process completions and depreciation."""
        completions = {}
        capacity_changes = {}

        for region in self.region_names:
            completed = self.pipeline.pop((region, t), 0.0)
            if preseed_completions and region in preseed_completions:
                completed += preseed_completions[region]
            completions[region] = completed

            self.pipeline_by_region[region] = max(0, self.pipeline_by_region[region] - completed)

            dep = DEPRECIATION_RATE_QTR * self.capacity[region]
            net = completed - dep
            self.capacity[region] = max(0, self.capacity[region] + net)
            capacity_changes[region] = net

        return {
            "completions": completions,
            "capacity": dict(self.capacity),
            "total_capacity": sum(self.capacity.values()),
            "capacity_changes": capacity_changes,
        }

    def total_pipeline(self) -> float:
        return sum(self.pipeline.values())

    def reset(self):
        self.capacity = {name: r.capacity_2025_mw for name, r in self.regions.items()}
        self.pipeline.clear()
        self.pipeline_by_region.clear()
