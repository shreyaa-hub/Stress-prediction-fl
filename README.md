# stress-prediction-fl

Code, processed data and result files for *Federated Learning for Smartphone-Based Stress Prediction: An Empirical Study of User Heterogeneity and Evaluation Protocols* (Singh & Natarajan B).

## Contents
| File | What it is |
|---|---|
| `stress_prediction_federated_learning.ipynb` | Full pipeline. Cells 0–12: original analysis. Cells R1–R6 (added Oct 2026): reconciliation analyses reported in the corrected paper |
| `labels.csv` | 1,193 labelled Stress-EMA responses (Scheme C: levels 2–3 = stressed, 4–5 = not stressed; level 1 excluded) |
| `features.csv` | 8 behavioural features over the causal 24-hour window before each response (after interval merging) |
| `fl_comparison.csv` | Per-client AUC for the four regimes (100 client × seed rows, cell 11) |
| `personal_results.csv` | Fully personalised model results (cell 10) |
| `fedavg_curve.npy` | FedAvg held-out AUC per seed × round (cell 11) |
| `requirements.txt` | Library versions under which all outputs reproduce exactly |

## Data
StudentLife (Wang et al., UbiComp 2014), loaded from the public Kaggle mirror `dartweichen/student-life`. Sensor inference codes follow the official documentation (https://studentlife.cs.dartmouth.edu/dataset.html): activity 0 stationary, 1 walking, 2 running, 3 unknown; audio 0 silence, 1 voice, 2 noise, 3 unknown.

## Reproducing
Cells 3–7 need the raw StudentLife files. Cells 8–12 and R1–R6 run from `features.csv` / `labels.csv` alone (place them in `/kaggle/working/` or change the paths). Use the pinned versions: with scikit-learn ≥ 1.8 the subject-level AUCs of cell 8 become 0.519 (LR) and 0.480 (RF) instead of 0.522 and 0.485.
