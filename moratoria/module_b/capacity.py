"""
Aggregate capacity model with pipeline tracking.

Tracks MW in construction pipeline by region, handles completions with
region-specific construction timelines, and applies depreciation.

Includes simple queue congestion: when a region's pipeline grows relative
to its installed capacity, construction takes longer (PJM queues, permitting
backlogs, labor/supply bottlenecks).

Theoretical grounding:
- Hsieh & Moretti (2019): localized supply constraints cause aggregate
  misallocation even when activity can in principle relocate
"""

from collections import defaultdict

from moratoria.config import DEPRECIATION_RATE_QTR, T_END
from moratoria.data.regions import REGIONS, RegionParams

# Congestion: 30% pipeline-to-capacity increase -> 30% longer construction
CONGESTION_SENSITIVITY = 0.3


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

    def add_to_pipeline(self, region: str, mw: float, t: int):
        """Add MW to a region's construction pipeline at time t."""
        if mw <= 0:
            return
        r = self.regions[region]
        base_time = r.interconnect_time_qtrs + r.build_time_qtrs

        # Simple congestion: pipeline/capacity ratio adds delay
        baseline_pipeline = self.capacity[region] * 0.15
        ratio = self.pipeline_by_region[region] / max(baseline_pipeline, 1)
        congestion = CONGESTION_SENSITIVITY * max(0, ratio - 1)
        construction_time = base_time * (1 + congestion)

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
