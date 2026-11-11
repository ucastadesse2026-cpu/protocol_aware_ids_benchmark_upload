# Amendment to the frozen analysis plan (fill in, commit BEFORE opening any result of the enlarged run)

Original freeze: commit `[hash]` dated `[date]` (config `[file]`, thresholds in its `preregistration:` block).
This amendment: commit `[hash]` dated `[date]`. Written before any output of Run 2 was inspected: [yes / no - be truthful].

## 1. Why this amendment exists (state it plainly)
Run 1 (first pass: UNSW-NB15 testing partition only, [72] NOS3 cycles, [108] satellite test windows, [45] attack windows)
was inspected before this amendment. It showed: (a) a small satellite test set; (b) terrestrial-trained models that flag
almost every NOS3 window as an attack on clean data; (c) a bug in the evaluation of defended Random Forests (the
augmentation surrogate was reused for evaluation). The changes below are therefore **data-dependent**: they were motivated
by Run 1 and are not a blind pre-registration. Run 2 is a replication under unchanged decision rules, not an independent prediction.

## 2. What changed since the freeze
- Data: NOS3 capture enlarged from [72] to [300] cycles ([N] windows; test split [n] windows, [m] attack); UNSW-NB15 official
  training partition for training and the official testing partition for the terrestrial test (`test_path`); CICIDS2017
  added ([rows cap]); NSL-KDD [included / not included].
- Code: independent evaluation surrogate for non-differentiable models; greedy-ccsds run on every defended model;
  new tables 6b and informational RQ3 row; clean->attack drop columns; conservative test-sampling interval column;
  parallel seeds. Code version: release `[zip name/date]`, commit `[hash]`.
- Anything else: [list, or "nothing"].

## 3. What did NOT change
Every threshold in `preregistration:` is identical to the frozen file: rq1_min_families=[4], rq2_tau_threshold=[0.5],
rq3_min_recovery=[0.5], rq3_max_time_multiple=[1.5], rq3_clean_acc_tol_pp=[1.0], rq4_tolerance_pp=[3.0].
The primary G uses PGD at the primary epsilon and pgd-ccsds (transfer-ccsds for non-differentiable models), as frozen.
Verify with `git diff <freeze-hash> -- configs` and paste the result: [empty / explain].

## 4. Status of the two runs in the paper
- Run 1: reported as the **first pass**, as measured, with the invalid defended-Random-Forest rows marked as such.
- Run 2: the **enlarged replication** under the same rules; conclusions are drawn from Run 2 and Run 1 side by side.
  If the two disagree, both are reported and the disagreement is discussed; neither is dropped.

## 5. Secondary analyses declared now (before Run 2 is inspected)
(a) G_protocol with the strongest available conformant attack per model (Table 6b); (b) paired PAA minus clean-augmentation
control differences with 95% intervals; (c) clean->attacked accuracy drop reported next to tau; (d) conservative
test-sampling half-widths; (e) [anything else you decide now].

## 6. Validity checks (pre-specified; they do not change a hypothesis rule, they decide whether a number is informative)
- A model whose clean satellite FPR is >= [0.90] is "degenerate": its G_protocol and tau are reported but flagged uninformative.
- A row whose accuracy under attack exceeds its clean accuracy is flagged "attack not adaptive" and excluded from robustness claims.
- The satellite test split must contain >= [200] attack windows and >= [200] benign windows; otherwise the satellite results are labelled underpowered.
- pgd-ccsds feasibility below [90]%: investigate with `tools/diagnose_feasibility.py` and report the cause.
- An RQ whose evidence fails a validity check is reported as "uninformative", not as "supported" or "not supported".

## 7. Reporting commitments
All four research questions are reported with their measured outcome whatever the direction; every later change is
listed as a **post-amendment deviation** with its reason; the paper states which analyses were pre-registered, which are
secondary-declared (Section 5), and which are exploratory.
