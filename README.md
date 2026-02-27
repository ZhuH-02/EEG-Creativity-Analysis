# EEG Creativity Analysis

Reproducible EEG pipeline for:

- Phase 1 data auditing and feature extraction
- Baseline multiclass classification for creativity-task phases

The baseline trainer is in `code/app.py` and uses local participant JSON files (`sub_*.json`).

## Project Status (as of 2026-02-27)

- Data audit script implemented and writing rubric-style outputs to `outputs/phase1_data_selection/`
- Baseline models implemented: Logistic Regression and XGBoost
- Subject-wise train/test split enabled with `GroupShuffleSplit`
- Recent runs saved under `results/baseline/logreg/` and `results/baseline/xgboost/`

## Data Description

### Source

This project uses the EEG creativity dataset referenced by DOI:

`10.17632/24yp3xp58b.1`

### Local data layout

Expected folder structure:

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

### What each file is used for

- `sub_XX.json`: primary source used by `code/app.py` for training/evaluation
- `P*.eeg`: raw EEG binary stream (used by the Phase 1 audit fallback loader)
- `P*.vhdr` and `P*.vmrk`: legacy BrainVision metadata files, not used by the current pipeline

### Data schema used by the baseline model

`sub_*.json` is expected to contain segment arrays keyed by phase text, for example:

- `1_idea generation`
- `2_rest`
- `3_idea evolution`

`app.py` maps text labels to canonical classes:

- `RST` (rest) -> `0`
- `IDG` (idea generation) -> `1`
- `IDE` (idea evolution) -> `2`
- `IDR` (idea rating) -> `3`

### Current configured participants and signal settings

From `code/config.py`:

- Participants: `P2` to `P10`
- Sampling rate: `500 Hz`
- Window size: `1000` samples (`2.0` seconds)
- Window overlap: `0.5` (50%)
- Test split: `0.3`

## Repository Layout

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
├── notebooks/
├── requirements.txt
└── EEG data/                      # local-only, gitignored
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Edit `code/config.py` to control:

- dataset path and participants: `DATA_DIR`, `PARTICIPANTS`
- signal/window parameters: `SAMPLING_RATE`, `WINDOW_SIZE`, `WINDOW_OVERLAP`
- split and reproducibility: `TEST_SIZE`, `RANDOM_SEED`
- model settings: `BASELINE_MODEL_CONFIG`, `XGBOOST_CONFIG`
- output options: `RESULTS_DIR`, `SAVE_PLOTS`, `PLOT_FORMAT`, `PLOT_DPI`

Optional keys supported by `app.py`:

- `MODEL_VARIANTS` (for example `["logreg", "xgboost"]`)
- `JSON_PHASE_TO_CANONICAL` (maps JSON text labels to canonical phase codes)

## How To Run

### 1. Phase 1 data audit

```powershell
python code/phase1_data_selection_audit.py
```

Outputs written to:

- `outputs/phase1_data_selection/features.csv`
- `outputs/phase1_data_selection/file_audit.json`
- `outputs/phase1_data_selection/missingness.csv`
- `outputs/phase1_data_selection/summary_stats.csv`
- `outputs/phase1_data_selection/duplicates.csv`
- `outputs/phase1_data_selection/outliers_summary.csv`
- `outputs/phase1_data_selection/example_rows.csv`
- `outputs/phase1_data_selection/plot_*.png`

### 2. Baseline training and evaluation

```powershell
python code/app.py
```

Current behavior:

- loads all configured `sub_*.json` files
- builds window-level time and frequency features
- splits by participant ID (subject-wise split)
- trains selected model variants
- saves metrics and plots per run

Outputs per run:

`results/baseline/<model_name>/<YYYYMMDD_HHMMSS>/`

Each run folder contains:

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `roc_curve.png` (binary-only case)

## Latest Baseline Snapshot (2026-02-27)

From:

- `results/baseline/logreg/20260227_170259/metrics.json`
- `results/baseline/xgboost/20260227_170300/metrics.json`

Metrics:

- Logistic Regression: accuracy `0.2948`, weighted F1 `0.3398`, weighted OvR ROC-AUC `0.6458`
- XGBoost: accuracy `0.4142`, weighted F1 `0.3819`, weighted OvR ROC-AUC `0.6037`

## Large Data and Version Control

Raw EEG files are intentionally not versioned in standard Git due to size.

Common ignored paths:

- `EEG data/`
- `.venv/`
- `__pycache__/`

Keep raw recordings local or manage them with a dedicated data workflow (for example Git LFS, DVC, or external object storage).

## Notes

- If `xgboost` is not installed, `app.py` skips XGBoost and continues.
- Subject-wise split requires at least two participants.
- `.vhdr` and `.vmrk` are no longer required for current training/evaluation runs.
