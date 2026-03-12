"""
Baseline growth curve for US data center capacity.

Models the total national investment flow (MW entering pipeline per quarter)
under a no-moratorium scenario, plus pre-seeded pipeline projects that were
already under construction before the simulation starts.

The key insight: reaching ~80 GW by 2030 from 25 GW in 2025 requires a
large pre-seeded pipeline (projects already under construction or in advanced
interconnection stages at start of 2026) plus aggressive new investment.

2030 capacity target: ~80 GW (operating demand)
The 2025-2026 consensus moved substantially upward from earlier projections:
- McKinsey (2025): >80 GW (need to "triple" from 25 GW)
- Goldman Sachs (Sept 2025): 80-90 GW US share of 122 GW global
- DOE (July 2024): ~75 GW implied (50 GW attributable to DCs on top of existing)
- BloombergNEF (Dec 2025): ~80 GW by 2030, revised up 36% from April estimate
- Bain (2025): up to 100 GW
- EPRI + Epoch AI (Aug 2025): 50+ GW for AI alone
Conservative floor: ~65 GW (Grid Strategies, adjusting for double-counting)

Sources:
- DOE (2024): data center electricity demand report
- McKinsey (2025): AI Power: Expanding DC Capacity to Meet Growing Demand
- Goldman Sachs (2025): AI to Drive 165% Increase in DC Power Demand by 2030
- JLL (Jan 2026): Global DC sector to nearly double to 200 GW
- BloombergNEF (Dec 2025): US DC power demand 106 GW by 2035
- Sightline Climate (2025): 30-50% pipeline slippage
"""

import numpy as np
from moratoria.config import (
    CAPACITY_2025_US_MW,
    CAPACITY_2030_TARGET_MW,
    T_END,
)


def compute_preseed_completions(t_end: int = T_END) -> np.ndarray:
    """
    Pre-seeded pipeline completions (MW coming online per quarter) from
    projects already under construction at the start of the simulation.

    These represent the ~20 GW of effective pipeline (after 35% slippage
    from ~30 GW under active construction) that will complete during the
    simulation window.

    Distribution: weighted toward mid-simulation (2027-2029) since most
    large projects take 2-4 years and were started in 2023-2025.
    """
    # Pre-seeded pipeline: reflects the massive 2024-2025 pipeline boom.
    # JLL (Jan 2026) reports 35+ GW under active construction globally; US is ~70%.
    # Additional projects in advanced interconnection stages (PJM, ERCOT queues).
    # After ~30% slippage, effective completions are ~50 GW over the simulation window.
    total_preseed_mw = 50_000

    # Completion profile: bell-shaped, peaking around Q6-Q10 (mid-2027 to mid-2028)
    t = np.arange(t_end)
    profile = np.exp(-0.5 * ((t - 8) / 5) ** 2)
    profile = profile / profile.sum()

    return total_preseed_mw * profile


def compute_baseline_investment_curve(
    t_end: int = T_END,
) -> np.ndarray:
    """
    Compute the quarterly gross investment (MW entering pipeline each quarter)
    for NEW projects initiated during the simulation window.

    This investment enters the pipeline and takes region-specific construction
    time to become capacity. It supplements the pre-seeded pipeline.

    The curve ramps as AI demand accelerates: ~3.0 GW/qtr → ~7.0 GW/qtr.
    Calibrated to industry capex trajectory: $150-200B annually by 2026-2027
    (Goldman Sachs, McKinsey), translating to ~3-7 GW/qtr of new pipeline entry.
    """
    t = np.arange(t_end)

    base_rate = 3000  # MW per quarter at t=0
    max_rate = 7000   # MW per quarter at peak
    midpoint = 8      # Ramp midpoint ~Q1 2028 (earlier: AI boom front-loaded)
    steepness = 0.35

    investment = base_rate + (max_rate - base_rate) / (
        1 + np.exp(-steepness * (t - midpoint))
    )

    return investment


def compute_capacity_trajectory(
    t_end: int = T_END,
    target_2030_mw: float = CAPACITY_2030_TARGET_MW,
    capacity_2025_mw: float = CAPACITY_2025_US_MW,
) -> np.ndarray:
    """
    Compute the desired US capacity trajectory (MW) for each quarter.

    This is the smooth target reference. Actual capacity will deviate
    under moratorium scenarios.
    """
    t = np.arange(t_end)
    k_min = capacity_2025_mw
    k_max = target_2030_mw
    t_mid = 10
    k_steep = 0.25

    return k_min + (k_max - k_min) / (1 + np.exp(-k_steep * (t - t_mid)))
