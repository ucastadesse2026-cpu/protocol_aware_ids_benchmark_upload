### Table 6b: G_protocol with the strongest conformant attack (EXPLORATORY)

| Model | Strongest conformant attack (most frequent) | G_protocol (worst case) pp |
|---|---|---|
| cnn | pgd_ccsds | -13.1 [-32.8, 6.6] |
| lstm | pgd_ccsds | -7.2 [-21.4, 7.0] |
| random_forest | transfer_ccsds | 0.0 [0.0, 0.0] |
| transformer | pgd_ccsds | -15.7 [-35.5, 4.1] |

EXPLORATORY post-hoc analysis, not pre-registered: the pre-registered G_protocol uses pgd-ccsds (transfer-ccsds for non-differentiable models). Here the conformant accuracy is the minimum over pgd-, transfer- and greedy-ccsds. Satellite test set: 1196 windows, the same for every seed, so intervals over seeds do not capture test-set sampling error (a binomial 95% half-width on a single accuracy near 50% is about 2.8 pp).
