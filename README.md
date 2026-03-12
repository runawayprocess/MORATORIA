# Data Center Moratoria Impact Model

A geographic displacement model estimating how US state-level data center moratoriums affect national compute capacity and AI development timelines.

## Running

```bash
python run_model.py --no-charts          # Tables only
python run_model.py --sensitivity        # Include parameter sensitivity sweeps
python run_model.py --validate           # Include logit allocation validation
python run_model.py --sobol              # Full Sobol global sensitivity analysis
python run_model.py --sobol --sobol-n 128  # Sobol with larger sample size
python run_model.py                      # Full analysis with charts
```

## Architecture

Three-module pipeline:

1. **Module A (Geographic Displacement)**: Multinomial logit (McFadden 1974) allocates investment across 13 regions. Moratoriums block investment, which is redirected to open regions with reallocation friction. Agglomeration dynamics (Krugman 1991) evolve over time.

2. **Module B (Capacity Pipeline)**: Tracks MW in construction pipelines by region with region-specific build times (4-16 quarter interconnection + 7-9 quarter construction). Includes queue congestion feedback and construction labor/equipment constraints.

3. **Module C (Compute Mapping)**: Maps quality-weighted capacity to hardware FLOP, applies exogenous algorithmic efficiency multiplier (Erdil & Besiroglu 2024), computes effective compute milestones and delay metrics.

## Key Parameters

| Parameter | Value | Source | Sensitivity Range |
|---|---|---|---|
| Investment elasticity | 0.25 | Baker, Bloom, Davis (2016); Gulen & Ion (2015) | 0.10 - 0.35 |
| Reallocation delay | 5 quarters | Site selection + permitting timeline | 3 - 8 quarters |
| Algo doubling time | 12 months | Erdil & Besiroglu (2024) | 8 - 18 months |
| Effective fungibility | 0.556 | Workload mix estimate | 0.4 - 0.7 |
| Fungibility price response | 0.5 | Endogenous scarcity feedback | 0.0 - 1.0 |
| Queue congestion sensitivity | 0.3 | PJM/ERCOT queue data (see note) | 0.1 - 0.6 |
| Queue congestion threshold | 0.15 | Pipeline-to-capacity onset ratio | 0.08 - 0.30 |
| Hardware improvement/qtr | 9.4% | Epoch AI GATE model | 6% - 12% |
| Agglomeration elasticity | 0.4 | Krugman (1991) | 0.2 - 0.7 |
| Logit temperature | 0.15 | Calibrated to regional distribution | 0.08 - 0.25 |

## Sensitivity Analysis

Run `--sobol` for a full Sobol global sensitivity analysis (Sobol 2001) across all 10 key parameters. This uses Saltelli sampling (SALib) to compute first-order (S1), total-order (ST), and second-order (S2) indices for peak delay, peak shortfall, and cumulative deficit.

**Key Sobol findings (N=64, 1408 model evaluations, 10 parameters):**
- **Peak delay** is dominated by `congestion_threshold` (S1=0.80), the pipeline-to-capacity ratio at which queue congestion begins. `algo_doubling_months` and `congestion_sensitivity` are secondary (ST~0.12 each).
- **Peak shortfall** is dominated by `congestion_threshold` (S1=0.94). All other parameters have near-zero first-order indices.
- **Cumulative deficit** is more distributed: `congestion_threshold` (S1=0.38, ST=0.47), `congestion_sensitivity` (ST=0.36), `fungibility_price_response` (ST=0.25), `investment_elasticity` (ST=0.15).
- **The two congestion parameters interact** (S2=0.13-0.20): the threshold determines *when* congestion activates, the sensitivity determines *how much* it hurts. Together they drive most model variance.
- **Logit temperature and agglomeration elasticity have near-zero influence** (ST < 0.07). ASC calibration absorbs their effect on base allocation.
- **Distribution**: delay p10=1.2wk, median=2.9wk, p90=4.2wk across all parameter combinations.

Run `--validate` to compare the logit allocation at t=0 against observed 2025 regional capacity shares. With ASC calibration, the model reproduces observed shares exactly.

## Logit Calibration

The displacement model uses alternative-specific constants (ASCs) computed analytically: `c_r = T * ln(obs_r) - f_r`. This ensures the logit reproduces observed 2025 regional capacity shares exactly. Feature weights then determine substitution patterns under moratorium shocks. Without ASCs, total absolute error is ~55% (NOVA: 32.4% observed vs 11.0% predicted). With ASCs, error is 0.0%.

## Known Limitations

### Removed: Algorithmic Endogeneity (ALGO_ENDOGENEITY)

We implemented and tested a parameter (ALGO_ENDOGENEITY = 0.15) representing the fraction of algorithmic progress that depends on hardware compute availability. The theory: when moratoriums reduce hardware availability, researchers have less compute for experiments, slightly slowing algorithmic progress.

**Result**: The effect was negligible. Under the broadest moratorium scenario, the algo multiplier at simulation end was 1692x vs 1722x for baseline (< 2% difference). This is because the hardware shortfall is small (~7%) and only 15% of algo progress was modeled as hardware-dependent. The parameter was removed for transparency. Algorithmic efficiency is now treated as fully exogenous and identical across scenarios.

### Static Compute Quality Scores

Regional `compute_quality` scores (NOVA=1.0, DFW=0.80, TX_OTHER=0.55, etc.) are fixed at 2025 levels. In reality, as investment flows to destination regions, fiber gets built, IX ecosystems develop, and quality converges upward. The quality degradation penalty reported by the model (+1.4% for ADT) may overstate the long-run effect because it doesn't capture infrastructure maturation in receiving regions.

### Persistent Shortfall is an Artifact of Exogenous Investment

Under the All Democratic Trifectas scenario, the capacity shortfall does not reequilibrate within the simulation window (through Q4 2036), even though moratoriums expire by Q2 2029. This is because the model uses an exogenous investment curve: the same S-curve (3,000-7,000 MW/qtr) regardless of market conditions. There is no "catch-up" mechanism.

In reality, once moratoriums expire, pent-up demand in affected regions would accelerate investment above the baseline rate. Capacity scarcity during the moratorium period would also raise colocation prices, attracting additional capital. A model with endogenous investment response would show faster recovery after moratorium expiry but also potentially larger immediate shortfalls (as the price signal takes time to propagate).

**The persistent shortfall should be interpreted as the cost of the moratorium period itself, not as a prediction that the gap never closes.** The cumulative compute deficit metric (which sums the shortfall over the full simulation) is more robust to this limitation than the end-of-simulation snapshot.

### No Physical Constraints on Destination Regions

When moratoriums redirect investment to fewer regions, the model applies interconnection queue congestion and construction labor congestion, but does not model:
- Power grid reliability limits (ERCOT capacity during summer/winter peaks)
- Generation adequacy (whether Texas can actually supply 15+ GW of new DC load)
- Transformer and switchgear supply chain bottlenecks
- Water stress in arid destination regions (Phoenix)

These physical limits would increase the effective capacity shortfall beyond what the model produces. The model's estimates should be read as a lower bound on delay.

### No Endogenous Investment Response

The baseline investment curve is exogenous: the same S-curve (3,000-7,000 MW/quarter) regardless of market conditions. In reality, capacity shortages raise DC prices, which should attract additional investment to open regions. Conversely, the model has no "catch-up" investment after moratoriums expire. A model with endogenous investment response would show faster recovery but also potentially larger immediate shortfalls (as the price signal takes time to propagate).

### Duration vs. Breadth Inversion in Cumulative Deficit

The Currently Considering scenario (narrower scope: NYC, VT/ME, GA) produces a *higher* cumulative compute deficit than All Democratic Trifectas (broader scope: all Dem trifecta states). This counterintuitive result is driven by three mechanisms:

1. **Moratorium duration**: CC's NYC moratorium is 3 years (to Q3 2029) and OTHER_BLUE runs to Q3 2030. ADT's broadest moratoria are 2-2.5 years (expire Q4 2028). Longer moratoriums create more persistent pipeline disruption.
2. **Destination blocking**: CC uniquely blocks Atlanta (a red-state destination region), reducing absorption capacity. ADT does not block any destination regions.
3. **Pipeline speed-up**: ADT redirects investment from slow-pipeline NOVA (PJM, 23-quarter build) to fast-pipeline DFW/TX (ERCOT, 11-quarter build). This briefly causes ADT capacity to *exceed* baseline around Q4 2031, partially offsetting later shortfalls.

The finding is robust to clipping (whether overshoot quarters offset deficit or not). Peak shortfall and peak delay metrics are unaffected and order correctly (ADT > CC). The implication for policy: moratorium *duration* matters more than moratorium *breadth* for cumulative compute impact.

### Static Moratorium Scenarios

Scenarios are fixed at enactment. No modeling of:
- Contagion (one state passing makes neighbors more likely)
- Federal preemption or national moratorium
- Political feedback (moratoriums causing job losses get repealed faster)
- Strategic firm behavior (lobbying, pre-committing investment)

### International Leakage Not Modeled

International regions (Persian Gulf, Canada, Mexico) are included as reference capacity but do not receive redirected US investment. Blocked investment is redistributed only among domestic US regions. In reality, some investment would leak abroad, but the magnitude depends on export controls, data sovereignty requirements, construction labor availability in destination countries, and corporate governance timelines that are difficult to parameterize credibly. The model's delay metrics reflect US-only displacement.
