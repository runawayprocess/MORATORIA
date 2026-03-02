"""
Moratorium scenario definitions.

Two main scenarios:
1. "Currently Considering": States with active, plausible moratorium proposals.
2. "All Democratic Trifectas": Every Democratic trifecta state (excl. Virginia) passes
   a moratorium. Virginia has a Dem trifecta but is unlikely to pass a moratorium
   given its economic dependence on the NoVA data center market.

Democratic trifecta states (as of early 2026, 16 total):
  CA, CO, CT, DE, HI, IL, ME, MD, MA, NJ, NM, NY, OR, RI, VA, WA

Notable non-trifectas that nonetheless have active moratorium proposals:
  VT (R governor), PA (R senate), GA (R trifecta, bipartisan bill),
  OK (R trifecta), SC (R trifecta), SD (R trifecta), MI (divided)

Sources:
  - Ballotpedia State Government Trifectas (2026)
  - MultiState: State Data Center Legislation in 2026
  - NY S.9144 / A.10141 (3-year moratorium, all new DCs)
  - GA SB 476 (1-year moratorium starting July 2026)
  - VT S.205 (moratorium through July 2030)
  - MD HB0120 (co-location requirement)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MoratoriumSpec:
    """Moratorium specification for a single region."""
    region: str
    start_t: int               # Quarter when moratorium begins (0 = Q1 2026)
    end_t: int                 # Quarter when moratorium ends (or T_END if indefinite)
    strength: float = 1.0      # 1.0 = full block; 0.0-1.0 = partial friction
    mw_threshold: float = 0.0  # Only blocks projects above this MW (0 = all)


@dataclass
class Scenario:
    """Complete scenario definition."""
    name: str
    description: str
    moratoria: list[MoratoriumSpec] = field(default_factory=list)
    queue_congestion_override: Optional[float] = None
    international_leakage_override: Optional[float] = None


# Q indices: 0=Q1'26, 1=Q2'26, 2=Q3'26, 3=Q4'26, 4=Q1'27, ...

SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        name="Baseline",
        description="No new moratoria. Business as usual with standard pipeline slippage.",
        moratoria=[],
    ),

    "currently_considering": Scenario(
        name="Currently Considering",
        description=(
            "States with active, plausible moratorium proposals pass them. "
            "NY S.9144 (3yr, all new DCs); VT S.205 (through 2030); "
            "GA SB 476 (1yr bipartisan); MD HB0120 (co-location requirement)."
        ),
        moratoria=[
            # NY S.9144: 3-year moratorium on all new DC permits
            # NY has strong momentum: Dem trifecta, high-profile sponsors (Sen. Krueger)
            MoratoriumSpec("NYC_NJ", start_t=2, end_t=14, strength=1.0),    # Q3 2026 -> Q2 2029

            # VT S.205 + ME proposals: small states, partial impact on OTHER_BLUE bucket
            # VT S.205 through July 2030; ME has Dem trifecta, likely to follow
            # These states are ~25% of OTHER_BLUE capacity
            MoratoriumSpec("OTHER_BLUE", start_t=3, end_t=18, strength=0.25),  # Q4 2026 -> Q2 2030

            # GA SB 476: 1-year moratorium starting July 2026 (bipartisan, R trifecta)
            MoratoriumSpec("ATL", start_t=2, end_t=6, strength=1.0),        # Q3 2026 -> Q2 2027
        ],
    ),

    "all_dem_trifectas": Scenario(
        name="All Democratic Trifectas",
        description=(
            "Every Democratic trifecta state (excl. VA) passes a moratorium. "
            "Covers NY, NJ, IL, CA, OR, WA, CO, ME, MD, CT, MA, RI. "
            "VA excluded (unlikely despite Dem trifecta). "
            "Assumes 2-year moratoria starting Q3 2026."
        ),
        moratoria=[
            # NY + NJ: both Dem trifectas, strong momentum
            MoratoriumSpec("NYC_NJ", start_t=2, end_t=14, strength=1.0),     # Q3 2026 -> Q2 2029 (NY 3yr)

            # IL: Dem supermajority in both chambers
            MoratoriumSpec("CHI", start_t=3, end_t=11, strength=1.0),        # Q4 2026 -> Q2 2028

            # CA + OR + WA: all Dem trifectas (entire West Coast bucket)
            # Strength 0.85: CA likely partial restrictions not full ban
            MoratoriumSpec("WEST_COAST", start_t=3, end_t=11, strength=0.85), # Q4 2026 -> Q2 2028

            # OTHER_BLUE: ME and CO are Dem trifectas (~25% of bucket).
            # Plus CT, MA, MD, RI are Dem trifectas with some DC capacity.
            # Effective strength ~0.55 reflecting trifecta share of bucket capacity.
            MoratoriumSpec("OTHER_BLUE", start_t=3, end_t=11, strength=0.55), # Q4 2026 -> Q2 2028

            # NOVA: VA excluded per user instruction.
            # MD (~15% of NOVA metro capacity) is a Dem trifecta with HB0120.
            MoratoriumSpec("NOVA", start_t=4, end_t=12, strength=0.15),      # Q1 2027 -> Q1 2028 (MD only)
        ],
    ),
}
