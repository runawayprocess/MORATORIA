"""
Baseline growth curve for US data center capacity.

Models the total national investment flow (MW entering pipeline per quarter)
under a no-moratorium scenario, plus pre-seeded pipeline projects that were
already under construction before the simulation starts.

The key insight: reaching 55 GW by 2030 from 25 GW in 2025 requires not
just new investment during the simulation, but also completing ~15-20 GW
of projects that were already in the pipeline at the start of 2026.

Sources:
- DOE (2024): data center electricity demand report
- S&P Global (2024): DC grid power demand to nearly triple by 2030
- JLL (2025): North America DC report: ~35 GW under construction YE 2025
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
    total_preseed_mw = 20_000

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

    The curve ramps as AI demand accelerates: ~1.5 GW/qtr → ~3.5 GW/qtr.
    """
    t = np.arange(t_end)

    base_rate = 1500  # MW per quarter at t=0
    max_rate = 3500   # MW per quarter at peak
    midpoint = 12     # Ramp midpoint ~Q1 2029
    steepness = 0.3

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
