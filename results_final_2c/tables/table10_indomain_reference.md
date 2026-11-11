### Table 10: in-domain reference, detectors trained on satellite windows (EXPLORATORY)

| Model | Acc clean % | FPR clean % | FNR clean % | Acc PGD % | FNR PGD % | Acc conformant % | FNR conformant % | Acc greedy % | Conformant success % | G_protocol in-domain pp | Informative? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | 73.33 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 0.0 [0.0, 0.0] | no (near-constant or chance-level detector) |
| lstm | 73.33 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 0.0 [0.0, 0.0] | no (near-constant or chance-level detector) |
| random_forest | 59.70 ± 0.00 | 66.77 ± 0.00 | 30.67 ± 0.00 | 24.50 ± 0.00 | 78.68 ± 0.00 | 62.46 ± 0.00 | 26.91 ± 0.00 | 17.14 ± 0.00 | 14.31 ± 0.00 | -38.0 [-38.0, -38.0] | no (near-constant or chance-level detector) |
| transformer | 73.33 ± 0.00 | 100.00 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 73.33 ± 0.00 | 0.00 ± 0.00 | 0.0 [0.0, 0.0] | no (near-constant or chance-level detector) |

EXPLORATORY: added after inspecting earlier runs. Features: extended, window 32 packets, label rule any, attack share of the training windows 62.3%. Satellite evaluation subset: 1196 windows, 877 attacks (an always-attack predictor scores 73.3%, an always-benign one 26.7%). Scaler fitted on satellite training windows.
