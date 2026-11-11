### Table 6: robustness gap (percentage points)

| Model | CCSDS attack | G | G_domain | G_protocol | G % | Clean cross-domain drop | Test-sampling ± pp (conservative) |
|---|---|---|---|---|---|---|---|
| cnn | pgd_ccsds | 19.01 ± 51.27 | 51.54 ± 5.27 | -32.53 ± 46.00 | 18.35 ± 69.43 | 43.25 ± 37.60 | 3.6 |
| lstm | pgd_ccsds | 42.69 ± 15.45 | 44.95 ± 12.25 | -2.26 ± 3.19 | 77.07 ± 14.55 | 42.60 ± 37.79 | 2.8 |
| random_forest | transfer_ccsds | 21.37 ± 43.71 | 21.37 ± 43.71 | 0.00 ± 0.00 | 25.63 ± 57.40 | 45.92 ± 36.09 | 4.0 |
| transformer | pgd_ccsds | 16.65 ± 58.84 | 47.50 ± 15.20 | -30.85 ± 43.63 | 8.50 ± 84.99 | 43.28 ± 37.70 | 3.6 |

G = G_domain + G_protocol (exact). Standard attack: pgd. Intervals over seeds reflect model variance only (the satellite test windows are identical for all seeds); the last column is a conservative binomial half-width for G_protocol from the number of satellite test windows (it ignores the pairing of the two accuracies).
