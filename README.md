# EEG Creativity Analysis

Reproducible EEG pipeline in Python + PyTorch for:

- Phase 1 data auditing and feature extraction
- Baseline multiclass classification of creativity-task phases

## Current Status (February 28, 2026)

- Baseline training is fully PyTorch (`torch_linear`, `torch_mlp`) in `code/app.py`
- Subject-wise split is enabled (participant-level grouping)
- Phase 1 audit outputs are generated under `outputs/phase1_data_selection/`
- Current baseline runs exist under:
  - `results/baseline/torch_linear/20260227_190331/`
  - `results/baseline/torch_mlp/20260227_190336/`

## Dataset

Source DOI:

`10.17632/24yp3xp58b.1`

Expected local layout:

```text
EEG data/
  Participant-2/
    sub_02.json
    P2.eeg
  Participant-3/
    sub_03.json
    P3.eeg
  ...
```

File usage:

- `sub_XX.json`: primary training/evaluation source for `code/app.py`
- `P*.eeg`: raw EEG stream (used by Phase 1 fallback audit loader)
- `P*.vhdr`, `P*.vmrk`: optional legacy metadata, not required for baseline training

## Label Mapping

`app.py` maps text labels to canonical classes:

- `RST` -> `0` (Rest)
- `IDG` -> `1` (Idea Generation)
- `IDE` -> `2` (Idea Evolution)
- `IDR` -> `3` (Idea Rating)

## Repository Structure

```text
.
├── code/
│   ├── app.py
│   ├── config.py
│   └── phase1_data_selection_audit.py
├── outputs/
│   └── phase1_data_selection/
├── results/
│   └── baseline/
│       ├── torch_linear/
│       └── torch_mlp/
├── notebooks/
├── requirements.txt
└── EEG data/                      # local-only, gitignored
```

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you see `ModuleNotFoundError`, ensure VS Code uses:

`<workspace>\.venv\Scripts\python.exe`

## Configuration

Main settings are in `code/config.py`:

- Data and participants: `DATA_DIR`, `PARTICIPANTS`
- Signal/windowing: `SAMPLING_RATE`, `WINDOW_SIZE`, `WINDOW_OVERLAP`
- Split and reproducibility: `TEST_SIZE`, `RANDOM_SEED`
- Model variants and hyperparameters:
  - `MODEL_VARIANTS`
  - `TORCH_LINEAR_CONFIG`
  - `TORCH_MLP_CONFIG`
- Output settings: `RESULTS_DIR`, `SAVE_PLOTS`, `PLOT_FORMAT`, `PLOT_DPI`

## Run

### 1. Phase 1 Audit

```powershell
python code/phase1_data_selection_audit.py
```

Outputs written to `outputs/phase1_data_selection/`:

- `features.csv`
- `file_audit.json`
- `missingness.csv`
- `summary_stats.csv`
- `duplicates.csv`
- `outliers_summary.csv`
- `example_rows.csv`
- `run_metadata.json`
- `plot_histograms.png`
- `plot_corr_heatmap.png`
- `plot_windows_per_participant.png`

### 2. Baseline PyTorch Training

```powershell
python code/app.py
```

Behavior:

- loads configured `sub_*.json` files
- extracts window-level time/frequency features
- performs subject-wise train/test split
- trains configured PyTorch variants
- saves run artifacts per model

Per-run output folder:

`results/baseline/<model_name>/<YYYYMMDD_HHMMSS>/`

Per-run files:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png` (binary-only)

## Latest Baseline Snapshot (February 27, 2026)

From:

- `results/baseline/torch_linear/20260227_190331/metrics.json`
- `results/baseline/torch_mlp/20260227_190336/metrics.json`

Metrics:

- `torch_linear`: accuracy `0.3968`, weighted F1 `0.3833`, weighted OvR ROC-AUC `0.6703`
- `torch_mlp`: accuracy `0.4019`, weighted F1 `0.3963`, weighted OvR ROC-AUC `0.6726`

## Notes

- `app.py` still accepts legacy aliases (`logreg`, `xgboost`) and maps them to PyTorch variants for compatibility.
- Subject-wise split requires at least 2 participants.
- Raw EEG data is intentionally not committed to Git.
