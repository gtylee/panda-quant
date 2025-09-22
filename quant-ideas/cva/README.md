# CVA Model Equivalence: HW1F vs G2++ (QuantLib-Python)

Demonstrates that for a one-directional linear rates portfolio, Hull–White (1F) and G2++ (2F) deliver indistinguishable unilateral CVA when both use the same discount curve and ATM vol level and credit/market are independent.

## What we demonstrate
- **Portfolio**: linear, level-dominant (payer par swaps 5Y/10Y/20Y; par bonds 2Y/5Y/10Y, 10mm each).
- **Market/Credit**: single flat curve 2.0%; flat hazard 150 bps; LGD 60%; no collateral; no WWR.
- **Models**: HW1F and G2++; both reuse the same curve; G2 configured with negligible second factor variance (η≈0) to match HW on level exposure.
- **Method**: simulate OU factors; rebuild discount curves pathwise via closed-form ZCBs; reprice instruments; EPE(t); CVA via discretized integral.

## Why this works
Exposure for level-only linear books is driven by the integral of the short rate (discount curve). With identical initial term structure and ATM level, HW and G2 produce nearly identical pathwise ZCBs, forwards, and annuities—hence similar EPE and CVA. The second factor in G2 contributes little here.

## Project layout
```
cva-model-equivalence/
  src/
    market.py     # curve & grid
    credit.py     # flat hazard, LGD
    portfolio.py  # par swaps & bonds
    models.py     # HW/G2 closed-form ZCBs, simulators, calibration fits
    exposure.py   # vectorized EPE engine (+bucket shift multipliers)
    cva.py        # CVA and reporting utils
    scenarios.py  # scenario runner & summary
    main.py       # CLI
  output/
    epe_profiles.csv
    cva_report.csv
    calibration_hw.csv
    calibration_g2.csv
    cva_attribution.csv
    cva_bucketed_delta.csv
    cva_bucketed_delta_test.csv
    scenario_summary.csv
  requirements.txt
  README.md
  README_math.md
```

## Install & run
```bash
cd cva-model-equivalence
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --paths 16384 --grid 1M_10Y_3M_30Y --seed 9
```

## Results: 1F vs 2F after calibration
- Across scenarios (see `output/scenario_summary.csv`), the CVA difference decomposes as:
  - **Calibration component** = CVA_G2(aligned to HW vols) − CVA_G2
  - **Model component** = CVA_HW − CVA_G2(aligned)
  - Total CVA diff = Calibration + Model
- Findings (examples with 1–2k paths, grid `3M_10Y`):
  - Swaps-only, ATM 20%, hazard 150 bps: total CVA diff ≈ −6.5k; calibration component ≈ −7.0k; model component ≈ +0.5k
  - Varying ATM (15–25%) and hazard (100–200 bps) shows the same pattern: the calibration term explains almost all of the difference; the model term is small (few hundred), confirming 1F ≈ 2F for level-dominant exposure.
  - When bonds are included (level-only NPV), both models agree almost perfectly (tiny model component), reinforcing the intuition.

## Bucketed CVA deltas
- We compute per-bucket 1 bp deltas using discount-ratio multipliers and validate vs a brute-force bumped curve; see `output/cva_bucketed_delta_test.csv`.
- The two methods agree closely across buckets; residual differences are numerical (grid/MC).

## Interpreting the differences
- The second factor in G2++ primarily adds slope/curvature risk. For linear books whose exposure is dominated by level (PC1), its incremental effect on EPE/CVA is negligible once both models are anchored to the same ATM level and term structure.
- Any large raw differences you might observe at low path counts are predominantly calibration artifacts (ATM anchoring mismatch). Aligning the volatility levels collapses the gap, leaving a small model remainder.

## Scenario runner
```bash
python -m src.scenarios  # or from a Python shell: run_scenarios()
```
Writes `output/scenario_summary.csv` with CVA_HW, CVA_G2, total difference, calibration vs model components, and EPE curve statistics for multiple cases.

## Math
See `README_math.md` for explicit formulas and the calibration/delta definitions used in this project.
