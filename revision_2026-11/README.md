# Revision analyses (R10, November 2026 — ICICNCT 2026 review, Paper ID 815)

Scripts behind every number added in the revised paper. Column `sleep_hours` in the CSV files is the feature called **longest dark** in the revised paper (the StudentLife `dark` stream records phone-in-dark-environment episodes of at least 1 h, not sleep).

| Script | Paper content |
|---|---|
| `extract_all_levels.py` | Features for all 2,167 parsed Stress-EMA responses (levels 1–5) → `results/revision_2026-11/features_all_levels.csv` |
| `sleep_proxy_check.py` | Sleep-EMA vs. longest dark / locked interval → `results/revision_2026-11/sleep_ema_night_window.csv` |
| `r10a_regimes_convergence_pfl.py` | Test-set sizes, pair-weighted AUC, same-16-client comparison, TOST / minimum detectable difference, PFL controls, convergence diagnostics; writes `results/central_curve.npy` (dashed line in Fig. 3) |
| `r10b_client_construction_ablation.py` | Table VII extra arms (random shards with real sizes; mixed shards with real sizes and label rates), drop-one-feature ablation, Holm correction |
| `r10c_label_schemes_attrition_nonlinear.py` | Attrition analysis, Table VIII (Schemes A/B/C), Table V MLP and gradient boosting |
| `r10d_sleep_proxy_testsize_mlp.py` | Sleep-proxy validation, larger-test-set subset, cluster bootstrap, federated MLP (Table VIII last row) |
| `r10e_same16_pvalues.py` | Overlap and Wilcoxon p for the same-16 column of Table VI; corrected p = 0.308 for the random hold-out |

Run from this folder with the pinned environment (scikit-learn 1.7.2, SciPy 1.15.3). `RESULTS_DIR` defaults to `../../results` (local project layout). In the GitHub repo, run from this folder with `RESULTS_DIR=..` (the result CSVs are in the repo root and in this folder). `outputs/` holds the printed results used in the paper. Regenerate Figs. 2–3 with `python ../make_figures.py ../../results OUT_DIR` after running `r10a`.
