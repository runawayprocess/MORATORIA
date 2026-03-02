"""
Core geographic displacement model.

Implements a multinomial logit (McFadden 1974) allocation of data center investment
across market regions, with moratorium shocks, reallocation delays, workload
fungibility filters, and Krugman (1991) agglomeration dynamics.

Sources:
- McFadden (1974): discrete choice / logit allocation
- Greenstone (2002): regulatory displacement of industrial activity
- Krugman (1991): agglomeration vs. centrifugal forces in location choice
- Dechezlepretre et al. (2022): investment leakage from environmental regulation
"""

import numpy as np
from collections import defaultdict

from moratoria.config import (
    LOGIT_WEIGHTS,
    LOGIT_TEMPERATURE,
    REALLOCATION_DELAY_QTRS,
    EFFECTIVE_FUNGIBILITY,
    AGGLOMERATION_BUILD_RATE,
    AGGLOMERATION_DECAY_RATE,
    AGGLOMERATION_ELASTICITY,
    T_END,
)
from moratoria.data.regions import REGIONS, RegionParams
from moratoria.data.scenarios import Scenario


# ---------------------------------------------------------------------------
# Agglomeration helpers (inlined from agglomeration.py)
# ---------------------------------------------------------------------------

def _agglomeration_contribution(score: float) -> float:
    """Diminishing returns: A^alpha where alpha < 1."""
    if score <= 0:
        return 0.0
    return score ** AGGLOMERATION_ELASTICITY


def _update_agglomeration(
    scores: dict[str, float],
    capacity: dict[str, float],
    capacity_changes: dict[str, float],
    moratoria_strength: dict[str, float],
) -> dict[str, float]:
    """
    Update agglomeration scores. Decay proportional to moratorium strength;
    build proportional to fractional capacity growth.
    """
    updated = {}
    for region, a in scores.items():
        strength = moratoria_strength.get(region, 0.0)
        decay = a * AGGLOMERATION_DECAY_RATE * strength

        build = 0.0
        k = capacity.get(region, 1.0)
        dk = capacity_changes.get(region, 0.0)
        if k > 0 and dk > 0:
            build = AGGLOMERATION_BUILD_RATE * (dk / k) * (1.0 - strength)

        updated[region] = max(0.0, min(1.0, a - decay + build))
    return updated


# ---------------------------------------------------------------------------
# Main displacement model
# ---------------------------------------------------------------------------

class DisplacementModel:
    """
    Geographic displacement model for data center investment.

    Allocates national investment across regions using multinomial logit,
    applies moratorium shocks, and tracks reallocation with friction.
    """

    def __init__(
        self,
        regions: dict[str, RegionParams] = None,
        reallocation_delay: int = REALLOCATION_DELAY_QTRS,
    ):
        self.regions = regions or REGIONS
        self.weights = LOGIT_WEIGHTS
        self.temperature = LOGIT_TEMPERATURE
        self.reallocation_delay = reallocation_delay
        self.region_names = list(self.regions.keys())

        # Agglomeration scores (mutable over time)
        self.agglomeration = {
            name: r.agglomeration_score_init for name, r in self.regions.items()
        }

        # Reallocation buffer: (destination_region, arrival_quarter) -> MW
        self.reallocation_buffer: dict[tuple[str, int], float] = defaultdict(float)

    def compute_scores(self, t: int) -> dict[str, float]:
        """Compute attractiveness score for each region."""
        scores = {}
        w = self.weights
        for name, r in self.regions.items():
            agg = _agglomeration_contribution(self.agglomeration[name])
            scores[name] = (
                w["grid"] * r.grid_speed_score
                + w["fiber"] * r.fiber_density_score
                + w["cost"] * r.cost_score
                + w["agglomeration"] * agg
                + w["policy"] * r.policy_score
                + w["water"] * r.water_score
                + w["labor"] * r.labor_score
            )
        return scores

    def allocate_baseline(self, t: int) -> dict[str, float]:
        """
        Baseline allocation shares via multinomial logit (domestic only).
        International regions get zero baseline share.
        """
        scores = self.compute_scores(t)
        domestic = {n: s for n, s in scores.items() if not self.regions[n].is_international}
        max_score = max(domestic.values())
        exp_scores = {n: np.exp((s - max_score) / self.temperature) for n, s in domestic.items()}
        total = sum(exp_scores.values())
        result = {n: v / total for n, v in exp_scores.items()}
        for n in self.region_names:
            if self.regions[n].is_international:
                result[n] = 0.0
        return result

    def get_moratorium_strength(self, region: str, t: int, scenario: Scenario) -> float:
        """Moratorium strength 0.0-1.0 for a region at time t."""
        max_strength = 0.0
        for m in scenario.moratoria:
            if m.region == region and m.start_t <= t < m.end_t:
                max_strength = max(max_strength, m.strength)
        return max_strength

    def allocate_with_moratoria(
        self, t: int, scenario: Scenario, total_investment_mw: float
    ) -> dict:
        """
        Allocate investment with moratorium shocks.

        Returns dict with allocation, blocked, to_international.
        """
        baseline_shares = self.allocate_baseline(t)

        # Apply moratorium shocks
        blocked = {}
        effective = {}
        for region in self.region_names:
            strength = self.get_moratorium_strength(region, t, scenario)
            base_mw = baseline_shares[region] * total_investment_mw
            blocked_mw = base_mw * strength
            blocked[region] = blocked_mw
            effective[region] = base_mw - blocked_mw

        total_blocked = sum(blocked.values())

        # Fungibility split: relocatable vs permanently lost
        relocatable_mw = total_blocked * EFFECTIVE_FUNGIBILITY
        non_relocatable_mw = total_blocked - relocatable_mw

        # Redirect destinations (all non-blocked regions compete via logit)
        arrival_t = t + self.reallocation_delay
        scores = self.compute_scores(t)
        redirect_scores = {}
        for region in self.region_names:
            strength = self.get_moratorium_strength(region, t, scenario)
            if strength < 1.0:
                redirect_scores[region] = scores[region] * (1.0 - strength)
            else:
                redirect_scores[region] = 0.0

        active = {r: s for r, s in redirect_scores.items() if s > 0}
        if active:
            max_s = max(active.values())
            exp_s = {r: np.exp((s - max_s) / self.temperature) if s > 0 else 0.0
                     for r, s in redirect_scores.items()}
            total_exp = sum(exp_s.values())
            redirect_shares = {r: v / total_exp for r, v in exp_s.items()} if total_exp > 0 else {r: 0.0 for r in self.region_names}
        else:
            redirect_shares = {r: 0.0 for r in self.region_names}

        to_international = sum(
            relocatable_mw * redirect_shares[r]
            for r in self.region_names if self.regions[r].is_international
        )

        # Add to reallocation buffer (delayed)
        for region, share in redirect_shares.items():
            if share > 0 and arrival_t <= T_END:
                self.reallocation_buffer[(region, arrival_t)] += relocatable_mw * share

        # Retrieve arrivals from buffer
        from_limbo = {}
        for region in self.region_names:
            key = (region, t)
            from_limbo[region] = self.reallocation_buffer.pop(key, 0.0)

        # Final allocation = remaining baseline + buffer arrivals
        allocation = {r: effective[r] + from_limbo[r] for r in self.region_names}

        return {
            "allocation": allocation,
            "blocked": blocked,
            "to_international": to_international,
        }

    def update_state(
        self,
        capacity: dict[str, float],
        capacity_changes: dict[str, float],
        moratoria_strength: dict[str, float],
    ):
        """Update agglomeration scores after a quarter's capacity changes."""
        self.agglomeration = _update_agglomeration(
            self.agglomeration, capacity, capacity_changes, moratoria_strength,
        )

    def reset(self):
        """Reset model state for a new simulation run."""
        self.agglomeration = {
            name: r.agglomeration_score_init for name, r in self.regions.items()
        }
        self.reallocation_buffer.clear()
