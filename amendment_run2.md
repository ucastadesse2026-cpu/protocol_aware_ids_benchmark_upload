# Amendment to the frozen analysis plan

Original freeze: commit `9f621fe` dated `2026-10-11` (config `configs/default.yaml`,
the first real run's pre-registered thresholds).
Run-1 config: `configs/nos3.yaml` (committed here; unchanged since the Run-1
capture of the real NOS3 stream).
This amendment: commit `PASTE_HASH_HERE` dated `2026-10-15`.
Written before any output of Run 2 was inspected: **yes**

## 1. Why this amendment exists
Run 1 (first pass: UNSW-NB15 testing partition only, 72 NOS3 cycles, 108
satellite test windows, 45 attack windows) was inspected before this
amendment. It showed: (a) a small satellite test set; (b) terrestrial-trained
models that flag almost every NOS3 window as an attack on clean data; (c) a
bug in the evaluation of defended Random Forests (the augmentation surrogate
was reused for evaluation). The changes below are data-dependent and not a
blind pre-registration. Run 2 is a replication under unchanged decision rules,
not an independent prediction.

## 2. What changed since the freeze
- Data: NOS3 capture enlarged from 72 to 300 cycles; CICIDS2017 added
  (100,000 rows cap); NSL-KDD not included in this pass.
- Code: independent evaluation surrogate for non-differentiable models;
  greedy-ccsds run on every defended model; new Table 6b and informational
  RQ3 row; clean->attack drop columns. Also fixed: `ts_in_header` semantics
  for cFS secondary headers (True) matching the observed packet structure.
  Code release: protocol_aware_ids_benchmark.zip (2026-10-15).
- Nothing else.

## 3. What did NOT change
Every threshold in `preregistration:` is identical to the frozen file at
`9f621fe`:
  rq1_min_families=4, rq2_tau_threshold=0.5, rq3_min_recovery=0.5,
  rq3_max_time_multiple=1.5, rq3_clean_acc_tol_pp=1.0, rq4_tolerance_pp=3.0.
Verified with `git show 9f621fe:configs/default.yaml` and
`grep preregistration configs/nos3.yaml`.

## 4. Status of the two runs in the paper
- Run 1: reported as the **first pass**, as measured, with the invalid
  defended-Random-Forest rows marked as such.
- Run 2: the **enlarged replication** under the same rules; conclusions are
  drawn from Run 2 with Run 1 side by side. If the two disagree, both are
  reported and the disagreement is discussed; neither is dropped.

## 5. Secondary analyses declared now (before Run 2 is inspected)
(a) G_protocol with the strongest available conformant attack per model
    (Table 6b);
(b) paired PAA minus clean-augmentation control differences with 95%
    intervals;
(c) clean->attacked accuracy drop reported next to tau;
(d) conservative test-sampling half-widths.

## 6. Validity checks (pre-specified; they do not change a hypothesis rule)
- A model whose clean satellite FPR is >= 0.90 is "degenerate": its G_protocol
  and tau are reported but flagged uninformative.
- A row whose accuracy under attack exceeds its clean accuracy is flagged
  "attack not adaptive" and excluded from robustness claims.
- The satellite test split must contain >= 200 attack windows and >= 200
  benign windows; otherwise the satellite results are labelled underpowered.
- pgd-ccsds feasibility below 90%: investigate with
  `tools/diagnose_feasibility.py` and report the cause.
- An RQ whose evidence fails a validity check is reported as
  "uninformative", not as "supported" or "not supported".

## 7. Reporting commitments
All four research questions are reported with their measured outcome whatever
the direction; every later change is listed as a post-amendment deviation with
its reason; the paper states which analyses were pre-registered, which are
secondary-declared (Section 5), and which are exploratory.
