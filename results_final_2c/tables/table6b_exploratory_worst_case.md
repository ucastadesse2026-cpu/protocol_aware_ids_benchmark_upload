### Table 6b: G_protocol with the strongest conformant attack (EXPLORATORY)

| Model | Strongest conformant attack (most frequent) | G_protocol (worst case) pp |
|---|---|---|
| cnn | pgd_ccsds | -32.5 [-445.8, 380.7] |
| lstm | pgd_ccsds | -1.5 [-20.6, 17.6] |
| random_forest | transfer_ccsds | 0.0 [0.0, 0.0] |
| transformer | pgd_ccsds | -30.9 [-422.9, 361.2] |

EXPLORATORY post-hoc analysis, not pre-registered: the pre-registered G_protocol uses pgd-ccsds (transfer-ccsds for non-differentiable models). Here the conformant accuracy is the minimum over pgd-, transfer- and greedy-ccsds. Satellite test set: 1196 windows, the same for every seed, so intervals over seeds do not capture test-set sampling error (a binomial 95% half-width on a single accuracy near 50% is about 2.8 pp).
