"""
Region definitions for 13 data center market regions (10 US + 3 international).

US capacity figures from JLL/CBRE North America Data Center Reports (H2 2025).
Grid speed scores derived from PJM/ERCOT/CAISO public interconnection queue data.
Fiber density from Lightyear/DatacenterHawk market analyses.
Cost scores from Cushman & Wakefield DC Development Cost Guide 2025.
Compute quality reflects fiber density, IX proximity, and interconnect quality
for AI training workloads (fixed at 2025-2026 levels per Critique 7).
International capacity from Structure Research / Arizton / CBRE reports (2025).
"""

from moratoria.config import RegionParams


REGIONS: dict[str, RegionParams] = {
    "NOVA": RegionParams(
        name="Northern Virginia / Maryland",
        short_name="NOVA",
        political_lean="blue",
        capacity_2025_mw=8500,     # ~34% of 25 GW national (world's largest DC market)
        grid_speed_score=0.20,      # PJM queue: 3-8yr average
        fiber_density_score=1.00,   # World's densest fiber/IX ecosystem
        cost_score=0.30,            # High land costs, but strong tax incentives (being phased out)
        agglomeration_score_init=1.00,  # Maximum agglomeration
        policy_score=0.40,          # Tax exemption under threat; SCC oversight increasing
        water_score=0.50,           # Moderate; 2B gal/yr in 2023 (63% increase from 2019)
        labor_score=0.80,           # Deep DC labor pool
        interconnect_time_qtrs=16,  # PJM average: ~4 years
        build_time_qtrs=7,          # ~21 months
        iso_rto="PJM",
        compute_quality=1.00,       # Best: Ashburn IX, densest fiber globally
    ),
    "NYC_NJ": RegionParams(
        name="New York / New Jersey",
        short_name="NYC_NJ",
        political_lean="blue",
        capacity_2025_mw=1600,
        grid_speed_score=0.25,
        fiber_density_score=0.85,   # Major IX hub (60 Hudson, 111 8th Ave)
        cost_score=0.15,            # Very high land/power costs
        agglomeration_score_init=0.60,
        policy_score=0.20,          # NY moratorium proposed (S.9144)
        water_score=0.60,
        labor_score=0.75,
        interconnect_time_qtrs=14,
        build_time_qtrs=8,
        iso_rto="PJM/NYISO",
        compute_quality=0.95,       # 60 Hudson, 111 8th Ave, major IX
    ),
    "CHI": RegionParams(
        name="Chicago / Illinois",
        short_name="CHI",
        political_lean="blue",
        capacity_2025_mw=1400,
        grid_speed_score=0.30,
        fiber_density_score=0.70,   # Major Midwest IX hub (350 E Cermak)
        cost_score=0.45,
        agglomeration_score_init=0.50,
        policy_score=0.50,
        water_score=0.80,           # Great Lakes water access
        labor_score=0.65,
        interconnect_time_qtrs=12,
        build_time_qtrs=7,
        iso_rto="PJM",
        compute_quality=0.85,       # 350 E Cermak, strong Midwest IX
    ),
    "WEST_COAST": RegionParams(
        name="West Coast (CA, OR, WA)",
        short_name="WEST",
        political_lean="blue",
        capacity_2025_mw=2800,      # CA ~1200 + OR ~600 + WA ~1000 (Quincy, Seattle)
        grid_speed_score=0.28,
        fiber_density_score=0.80,   # Major tech hub fiber; Quincy WA is a fiber hub
        cost_score=0.25,            # Very high (CA/SV) to moderate (OR/WA)
        agglomeration_score_init=0.65,
        policy_score=0.35,          # OR has existing DC restrictions; CA/WA regulatory
        water_score=0.55,
        labor_score=0.85,           # Strong tech labor market
        interconnect_time_qtrs=10,
        build_time_qtrs=7,
        iso_rto="CAISO/BPA",
        compute_quality=0.88,       # Major tech hubs, good fiber, some geographic dispersion
    ),
    "OTHER_BLUE": RegionParams(
        name="Other Blue/Purple States (CT, MA, MN, WI, VT, ME, CO, PA, MD)",
        short_name="OTH_BLUE",
        political_lean="blue",
        capacity_2025_mw=1500,      # Includes CT/MA/MD capacity not in NOVA/NYC buckets
        grid_speed_score=0.35,
        fiber_density_score=0.35,
        cost_score=0.55,
        agglomeration_score_init=0.22,
        policy_score=0.30,          # Multiple moratoria proposed
        water_score=0.70,
        labor_score=0.42,
        interconnect_time_qtrs=10,
        build_time_qtrs=8,
        iso_rto="Various",
        compute_quality=0.70,       # Mixed; some good fiber (MA/CT), some sparse
    ),
    "DFW": RegionParams(
        name="Dallas-Fort Worth",
        short_name="DFW",
        political_lean="red",
        capacity_2025_mw=3500,
        grid_speed_score=0.65,      # ERCOT: 6-12mo (but growing)
        fiber_density_score=0.65,   # Growing IX ecosystem (Infomart)
        cost_score=0.70,            # Competitive land + power
        agglomeration_score_init=0.70,
        policy_score=0.80,          # TX very pro-DC
        water_score=0.40,           # Water stress in North TX
        labor_score=0.60,
        interconnect_time_qtrs=4,   # ERCOT base
        build_time_qtrs=7,
        iso_rto="ERCOT",
        compute_quality=0.80,       # Growing IX, Infomart hub
    ),
    "PHX": RegionParams(
        name="Phoenix",
        short_name="PHX",
        political_lean="swing",
        capacity_2025_mw=2000,
        grid_speed_score=0.50,
        fiber_density_score=0.50,
        cost_score=0.65,
        agglomeration_score_init=0.55,
        policy_score=0.75,          # AZ pro-DC incentives
        water_score=0.25,           # Severe water constraints
        labor_score=0.50,
        interconnect_time_qtrs=6,
        build_time_qtrs=7,
        iso_rto="WestConnect",
        compute_quality=0.72,       # Developing IX ecosystem
    ),
    "ATL": RegionParams(
        name="Atlanta / Georgia",
        short_name="ATL",
        political_lean="swing",
        capacity_2025_mw=1500,
        grid_speed_score=0.45,
        fiber_density_score=0.55,
        cost_score=0.60,
        agglomeration_score_init=0.45,
        policy_score=0.50,          # Bipartisan moratorium bill (SB 476)
        water_score=0.55,
        labor_score=0.55,
        interconnect_time_qtrs=8,
        build_time_qtrs=7,
        iso_rto="SERC",
        compute_quality=0.75,       # 56 Marietta, decent SE IX
    ),
    "TX_OTHER": RegionParams(
        name="Other Texas (Houston, San Antonio, West TX)",
        short_name="TX_OTH",
        political_lean="red",
        capacity_2025_mw=1200,
        grid_speed_score=0.60,
        fiber_density_score=0.35,
        cost_score=0.75,            # Cheap land, abundant renewables (W. TX)
        agglomeration_score_init=0.25,
        policy_score=0.80,
        water_score=0.35,           # Variable; W. TX is dry
        labor_score=0.45,
        interconnect_time_qtrs=4,
        build_time_qtrs=8,
        iso_rto="ERCOT",
        compute_quality=0.55,       # Sparse fiber in W. TX; Houston better
    ),
    "OTHER_RED": RegionParams(
        name="Other Red States (UT, NV, ID, SC, TN, LA, MS, AL)",
        short_name="OTH_RED",
        political_lean="red",
        capacity_2025_mw=2250,
        grid_speed_score=0.48,
        fiber_density_score=0.27,
        cost_score=0.68,
        agglomeration_score_init=0.17,
        policy_score=0.70,
        water_score=0.55,
        labor_score=0.33,
        interconnect_time_qtrs=6,
        build_time_qtrs=8,
        iso_rto="Various",
        compute_quality=0.55,       # Mixed; some remote locations
    ),
    "PERSIAN_GULF": RegionParams(
        name="Persian Gulf (UAE, Saudi Arabia, Qatar)",
        short_name="P_GULF",
        political_lean="international",
        capacity_2025_mw=800,
        grid_speed_score=0.75,         # State-backed grid buildout
        fiber_density_score=0.40,      # Growing: Jeddah/Fujairah subsea cable hubs
        cost_score=0.85,               # Very cheap power (nat gas/solar), subsidized land
        agglomeration_score_init=0.15,
        policy_score=0.95,             # Sovereign wealth fund backing; fast-track permits
        water_score=0.30,              # Desalination available but costly
        labor_score=0.20,              # Limited local talent; expat-dependent
        interconnect_time_qtrs=3,
        build_time_qtrs=8,
        iso_rto="National",
        compute_quality=0.45,          # Limited fiber, US export controls on advanced GPUs
        is_international=True,
    ),
    "CANADA": RegionParams(
        name="Canada (Quebec, Ontario, BC)",
        short_name="CAN",
        political_lean="international",
        capacity_2025_mw=1000,         # Montreal, Toronto, Vancouver markets
        grid_speed_score=0.60,         # Quebec Hydro fast-track; Ontario moderate
        fiber_density_score=0.65,      # Good fiber to US NE; Montreal IX hub
        cost_score=0.70,               # Quebec hydro very cheap; Ontario moderate
        agglomeration_score_init=0.35, # Existing DC market (Montreal esp.)
        policy_score=0.70,             # Generally supportive; some provincial variation
        water_score=0.85,              # Abundant (Great Lakes + Quebec rivers)
        labor_score=0.55,              # Good tech talent in Montreal/Toronto/Vancouver
        interconnect_time_qtrs=5,
        build_time_qtrs=7,
        iso_rto="IESO/HQ",
        compute_quality=0.80,          # Good interconnect to US NE; Montreal IX
        is_international=True,
    ),
    "MEXICO": RegionParams(
        name="Mexico (Queretaro, Monterrey, CDMX)",
        short_name="MEX",
        political_lean="international",
        capacity_2025_mw=400,          # Queretaro, Monterrey, Mexico City
        grid_speed_score=0.45,         # CFE grid; moderate challenges
        fiber_density_score=0.35,      # Growing; Queretaro improving
        cost_score=0.80,               # Cheap labor and land
        agglomeration_score_init=0.10, # Nascent DC market
        policy_score=0.55,             # Generally supportive but some policy uncertainty
        water_score=0.40,              # Water stress in northern Mexico
        labor_score=0.30,              # Limited specialized DC labor
        interconnect_time_qtrs=6,
        build_time_qtrs=9,
        iso_rto="CENACE",
        compute_quality=0.55,          # Emerging fiber; limited IX
        is_international=True,
    ),
}


def total_capacity_2025() -> float:
    """Sum of all US regional capacities (MW). Should approximate 25 GW."""
    return sum(r.capacity_2025_mw for r in REGIONS.values() if not r.is_international)


def us_regions() -> list[str]:
    return [k for k, v in REGIONS.items() if not v.is_international]


def international_regions() -> list[str]:
    return [k for k, v in REGIONS.items() if v.is_international]


def blue_regions() -> list[str]:
    return [k for k, v in REGIONS.items() if v.political_lean == "blue"]


def red_regions() -> list[str]:
    return [k for k, v in REGIONS.items() if v.political_lean == "red"]


def swing_regions() -> list[str]:
    return [k for k, v in REGIONS.items() if v.political_lean == "swing"]
