### Table 4: standard attacks at eps=0.05

| Model | Attack | Mode | Acc T % | Acc S % | FNR S % | Validator rejection (S) % |
|---|---|---|---|---|---|---|
| cnn | autoattack | whitebox | 68.83 ± 18.05 | 17.39 ± 13.13 | 94.47 ± 7.82 | 50.00 ± 70.71 |
| cnn | fgsm | whitebox | 63.79 ± 25.43 | 14.09 ± 17.80 | 98.97 ± 1.45 | 100.00 ± 0.00 |
| cnn | pgd | whitebox | 69.01 ± 18.28 | 17.47 ± 13.01 | 94.36 ± 7.98 | 100.00 ± 0.00 |
| lstm | autoattack | whitebox | 56.69 ± 32.00 | 13.38 ± 18.80 | 100.00 ± 0.00 | 47.95 ± 67.81 |
| lstm | fgsm | whitebox | 61.37 ± 28.94 | 48.54 ± 30.92 | 52.05 ± 67.81 | 100.00 ± 0.00 |
| lstm | pgd | whitebox | 58.33 ± 31.05 | 13.38 ± 18.80 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| random_forest | fgsm | transfer | 67.26 ± 16.61 | 50.00 ± 32.99 | 50.00 ± 70.71 | 100.00 ± 0.00 |
| random_forest | pgd | transfer | 71.37 ± 10.72 | 50.00 ± 32.99 | 50.00 ± 70.71 | 100.00 ± 0.00 |
| transformer | autoattack | whitebox | 66.45 ± 26.02 | 18.44 ± 11.65 | 93.04 ± 9.84 | 50.00 ± 70.71 |
| transformer | fgsm | whitebox | 71.04 ± 27.16 | 16.35 ± 14.60 | 95.90 ± 5.81 | 100.00 ± 0.00 |
| transformer | pgd | whitebox | 66.65 ± 25.85 | 19.15 ± 10.64 | 92.08 ± 11.21 | 100.00 ± 0.00 |

Validator rejection is a necessary-condition proxy for feature-space examples (lower bound).
