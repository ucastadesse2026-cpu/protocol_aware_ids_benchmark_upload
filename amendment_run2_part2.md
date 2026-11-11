# Amendment 2 (Run 2 / Stage B)

Parent amendment: `amendment-1` tag (commit `ab3b344`).
Freeze commit: `9f621fe`.
This amendment written before Run-2B outputs were inspected: **yes**

## 1. Why this amendment exists

The separability diagnostic (`tools/diagnose_separability.py --seed 1`)
shows no design over (feature_view, window_size, label_rule) reaches
validation balanced accuracy above 0.58. Best: sat_aug/window=8/rule=any
at 0.576 balanced accuracy, 0.619 AUC. Per-feature Cohen's d shows that
only three attack classes (large_payload |d|~3-5.6, seq_replay |d|=0.82,
apid_spoof |d|=0.69) are detectable; six others are at |d|~0.20.

Therefore Run-2B adds CICIDS2017 to the terrestrial datasets without
changing the model, attack, or feature design, to make Amendment-1's
claim truthful. No design change is being made because the diagnostic
shows no design would help.

## 2. What changed since Amendment-1

- `configs/nos3.yaml`: `datasets: [unsw_nb15]` -> `[unsw_nb15, cicids2017]`
- `configs/nos3.yaml`: `cicids2017.max_rows: 100000` -> `500000`
- No change to model code, attack code, feature code, or satellite data.

## 3. What did NOT change

- `preregistration:` thresholds (verified with tools/check_freeze.py)
- Feature view, window size, label rule
- NOS3 capture (1200-cycle, 191,248 packets)
- Federated config (reduced grid: K=5, alpha=0.5, 10 rounds, 2 variants)

## 4. Status of runs

- Run 1: first pass (72 cycles, UNSW only)
- Run 2: enlarged (1200 cycles, UNSW only)
- Run 2B (this run): 1200 cycles, UNSW + CICIDS2017. Design unchanged.
- Separability diagnostic (exploratory): reported in this amendment.
- In-domain reference (Table 10, exploratory): NOT rerun.

## 5. Validity checks

All Amendment-1 validity checks remain in force. Any model whose clean
satellite FPR >= 0.90 is flagged "degenerate", its G_protocol reported
uninformative.

## 6. Reporting commitments

Run-2B results are the primary evidence; Run-1 and Run-2 are ablations.
The separability diagnostic is reported as the explanation for why the
four pre-registered RQs are uninformative.

## 8. Pre-Run-2C data correction (added before the run)

The initial Run-2B was aborted after 30 minutes (before producing any
tables) because it used the 82,332-row UNSW-NB15 testing partition for
terrestrial training. We replaced this with the official 175,341-row
UNSW-NB15 training partition and restarted the pipeline. No results from
the aborted run are reported.

## 9. Compute path: GPU deadlock and CPU fallback (Run-2C)

Seeds 1 and 2 completed on GPU 2. Seeds 3 and 4 hung at `run_defenses` due to a CUDA runtime deadlock (host CUDA 13.0 vs older CUDA libs in the container). PyTorch was reinstalled as CPU-only (`torch==2.x+cpu`), and seeds 3, 4, 5 were rerun on CPU. No model, hyperparameter, or dataset differences; only the compute hardware changed. Aggregated metrics across seeds remain valid.
