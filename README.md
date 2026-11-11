# protocol_aware_ids_benchmark

Benchmark code for *“Protocol-Aware Adversarial Robustness of Satellite Intrusion Detection Systems: A Public-Dataset
Benchmark Against CCSDS-Conformant Attacks”*. It trains six IDS families on terrestrial data, attacks them with
standard and **CCSDS-conformant** (packet-level, C1–C8-validated) attacks, decomposes the robustness gap
`G = G_domain + G_protocol`, evaluates two defenses and a federated extension, and builds the paper's tables/figures
**from measured CSVs only**.

## Status — read this first
| Part | Verified in the authoring sandbox? |
|---|---|
| CCSDS parser, C1–C8 validator, feature schema, NumPy MLP, Random Forest, FedAvg, standard attacks (own FGSM/PGD), pgd/transfer/greedy-CCSDS attacks, AT, PAA, metrics, `make_tables.py`, all five experiment scripts | **Yes** — unit tests + end-to-end smoke test pass (see `tests/`). |
| PyTorch models (CNN, LSTM, Transformer), GAN-augmented model, torch FedAvg backbone, ART AutoAttack wrapper | **Executed once by the user (Windows): all passed except the AutoAttack wrapper, fixed afterwards and not re-run** (no PyTorch/ART in the sandbox). Run `python -m unittest tests.test_torch_models -v` first and expect to fix small bugs. |
| Real datasets (UNSW-NB15, CICIDS2017, NSL-KDD), UNSW-IoTSAT, CuCD-ID, NOS3/cFS | **No.** Not downloaded; the last three could not be verified at all (see `data/README.md`). |

The synthetic data and `configs/smoke.yaml` exist **only** to test plumbing. Every CSV row records `terr_source` /
`sat_source`; if any row is synthetic, `make_tables.py` stamps every table "SYNTHETIC-DATA BANNER" and every figure with a
watermark. Do not report such numbers.

## Setup
**Start here: `docs/COMPLETE_GUIDE.md`** (NOS3 on a command-line server: `docs/NOS3_HEADLESS_GUIDE.md`) - every step from a fresh Anaconda Prompt (environment, data, NOS3/cFS capture, experiments, results, writing and releasing the paper).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch + ART are only needed for the neural/GAN/AutoAttack parts
python -m unittest discover -s tests -v  # torch tests skip automatically when torch is absent
```
## Quick check (no PyTorch, no datasets): synthetic smoke run
```bash
for s in run_baseline run_standard_attacks run_ccsds_attacks run_defenses run_federated; do
  python experiments/$s.py --config configs/smoke.yaml; done
python make_tables.py --config configs/smoke.yaml      # -> results_smoke/tables, results_smoke/figures
```
## Real experiments
1. Get the data (`data/README.md`), edit `configs/default.yaml` (paths, `data.satellite`, **mission `profile`**, and
   fix the `preregistration` thresholds *before* the first run; commit the file).
2. Run, in order (all accept `--config configs/default.yaml` or a per-dataset config, `--seeds 1 2 3 4 5`, `--no-cache`):
```bash
python experiments/run_baseline.py          --config configs/default.yaml
python experiments/run_standard_attacks.py  --config configs/default.yaml
python experiments/run_ccsds_attacks.py     --config configs/default.yaml
python experiments/run_defenses.py          --config configs/default.yaml
python experiments/run_federated.py         --config configs/default.yaml
python make_tables.py                       --config configs/default.yaml
```
Outputs: `results/tables/*.csv|*.md` (Tables 3–9, plus `gap_per_seed.csv`) and `results/figures/fig3–5*.png`.
Paper Table 1/2, Figures 1–2 are descriptive (Table 2 = `configs/default.yaml → models`).

## Layout
`data/` parser, schema, preprocessing, synthetic generators · `models/` six families + NumPy MLP (surrogate) ·
`attacks/` standard, conformant, validator · `defenses/` AT, PAA · `evaluation/` metrics, gap, τ, onboard ·
`experiments/` one script per phase · `make_tables.py` · `tests/` · `configs/`.

## Design decisions the paper must state (they differ from or refine the manuscript draft)
1. **Threat model:** only attack-class samples are perturbed; accuracy under attack = accuracy over the whole test set.
2. **Common features (Fc):** terrestrial flows and CCSDS windows are mapped to the same duration/bytes/packet features
   (`core`: 4, `extended`: 8). NSL-KDD supports `core` only. Direction: **TC = forward, TM = backward** (assumption).
   Information specific to each domain is discarded — a limitation.
3. **LSTM** treats the feature vector as a token sequence (no windows of consecutive flows).
4. **Conformant attacker** controls only TC packets (configurable), edits padding length and inter-packet timing
   (and APID in greedy-CCSDS); with the common features only these two can change Fc. Replay is out of scope (it violates
   C3/C7 by definition). A candidate that fails C1–C8 (no *new* violations allowed) is infeasible → attack counted as failed.
5. **pgd-ccsds** = gradient chained through a finite-difference Jacobian of the feature map; budget B = packets edited.
   **transfer-ccsds** = pgd-ccsds on the surrogate (`attacks.surrogate`, default NumPy MLP) scored on the target.
6. **Standard-attack realizability** (Table 4 last column) is a *necessary-condition proxy* on features (lower bound).
7. **AutoAttack** = ART's APGD-CE component only (Square is image-only, APGD-DLR needs >2 classes); say "APGD-CE (ART)" in the paper.
8. **Random Forest:** attacked by transfer/greedy; "adversarial training" = one-shot augmentation with surrogate examples.
9. **PAA** is generated once (no per-epoch refresh). `paa_clean_control` (same satellite windows, unperturbed) is
   mandatory to separate *domain exposure* from *adversarial content*; Table 7/9 report both.
10. **Gap definition:** `G` uses `gap.standard_attack` (PGD at `primary_eps`) and the first available of
    `gap.ccsds_attack_preference`; ε is in min-max-scaled units; τ is "n/a" unless terrestrial robustness gain ≥ 1 pp.
11. **Federated:** Dirichlet label skew; nodes with a single class skip perturbation; FedAvg is implemented from scratch.
12. **Latency** is a single-thread host-CPU proxy, not flight hardware.

## Reproducibility
Seeds 1–5 (`--seeds`), splits by seed (satellite: chronological), config copied into your run logs, model cache in
`results/cache` (weights; delete or `--no-cache` to retrain). Licence: choose before publishing (none is set here).
