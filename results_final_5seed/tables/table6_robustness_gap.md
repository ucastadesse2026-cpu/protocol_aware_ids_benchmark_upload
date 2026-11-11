### Table 6: robustness gap (percentage points)

| Model | CCSDS attack | G | G_domain | G_protocol | G % | Clean cross-domain drop | Test-sampling ± pp (conservative) |
|---|---|---|---|---|---|---|---|
| cnn | pgd_ccsds | 12.92 ± 45.14 | 25.98 ± 41.90 | -13.06 ± 27.53 | -2.05 ± 75.62 | 43.40 ± 27.57 | 3.9 |
| lstm | pgd_ccsds | 15.20 ± 46.63 | 22.79 ± 42.76 | -7.59 ± 20.56 | -2.81 ± 95.29 | 42.82 ± 27.97 | 3.9 |
| random_forest | transfer_ccsds | 23.06 ± 30.77 | 23.06 ± 30.77 | 0.00 ± 0.00 | 28.50 ± 39.84 | 45.95 ± 26.83 | 4.0 |
| transformer | pgd_ccsds | 16.23 ± 44.79 | 31.93 ± 33.26 | -15.69 ± 27.72 | 4.64 ± 71.00 | 43.28 ± 27.97 | 3.9 |

G = G_domain + G_protocol (exact). Standard attack: pgd. Intervals over seeds reflect model variance only (the satellite test windows are identical for all seeds); the last column is a conservative binomial half-width for G_protocol from the number of satellite test windows (it ignores the pairing of the two accuracies).
