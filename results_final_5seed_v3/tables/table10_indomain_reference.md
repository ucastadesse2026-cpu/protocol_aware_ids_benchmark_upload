### Table 10: in-domain reference, detectors trained on satellite windows (EXPLORATORY)

| Model | Acc clean % | FPR clean % | FNR clean % | Acc PGD % | FNR PGD % | Acc conformant % | FNR conformant % | Acc greedy % | Conformant success % | G_protocol in-domain pp | Informative? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 73.33 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 0.0 [0.0, 0.0] | no (near-constant or chance-level detector) |
| lstm | 73.33 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 0.0 [0.0, 0.0] | no (near-constant or chance-level detector) |
| random_forest | 59.58 ± 0.29 | 66.65 ± 0.26 | 30.88 ± 0.49 | 25.45 ± 13.97 | 77.42 ± 18.96 | 26.24 ± 19.09 | 76.35 ± 26.06 | 17.17 ± 0.07 | 69.79 ± 29.24 | -0.8 [-18.0, 16.4] | no (near-constant or chance-level detector) |
| transformer | 73.33 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 58.90 ± 30.42 | 0.00 ± 0.00 | 0.0 [0.0, 0.0] | no (near-constant or chance-level detector) |

EXPLORATORY: added after inspecting earlier runs. Features: extended, window 32 packets, label rule any, attack share of the training windows 62.3%. Satellite evaluation subset: 1196 windows, 877 attacks (an always-attack predictor scores 73.3%, an always-benign one 26.7%). Scaler fitted on satellite training windows.
