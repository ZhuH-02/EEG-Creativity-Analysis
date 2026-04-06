# Project Structure And Figure Guide

## 1. Project Goal

This project builds an EEG phase-classification system for creativity-task analysis.

The full workflow now covers:

- Phase 1: data audit and feature extraction
- Milestone 2: training and baseline evaluation
- Milestone 3: failure checks, robustness, monitoring, and adaptation
- Milestone 4: continual learning, HITL, active learning, end-to-end integration, and final reporting

The main label mapping used across the project is:

- `RST` = Rest
- `IDG` = Idea Generation
- `IDE` = Idea Evolution
- `IDR` = Idea Rating

## 2. Top-Level Directory Structure

### `code/`

This is the main source directory.

- `config.py`
  Central configuration file for data paths, participants, signal settings, and model defaults.

- `app.py`
  Legacy baseline entrypoint. Loads EEG JSON windows, extracts features, trains baseline linear and MLP models, and saves results under `results/baseline/`.

- `train_pipeline.py`
  Main Milestone 2 training pipeline. Supports participant-wise `train/val/test` split, early stopping, richer metrics, checkpoints, and efficiency profiling.

- `analysis_utils.py`
  Shared utility layer for loading saved runs, reconstructing datasets, evaluating models, profiling inference, and saving outputs.

- `phase1_data_selection_audit.py`
  Phase 1 data audit script. Produces feature summaries, missingness statistics, outlier checks, and audit plots.

- `failure_checks.py`
  Milestone 3 anticipated failure analysis and stress testing.

- `robustness_eval.py`
  Milestone 3 robustness and calibration evaluation, including FGSM and confidence diagnostics.

- `monitoring_sim.py`
  Milestone 3 offline monitoring simulation for drift and alerting.

- `adaptation_eval.py`
  Milestone 3 drift adaptation experiment using warm-start fine-tuning.

- `continual_learning.py`
  Milestone 4 replay-based continual learning module.

- `hitl_active_learning.py`
  Milestone 4 HITL trigger and human feedback simulation module.

- `active_learning_eval.py`
  Milestone 4 active learning strategy comparison module.

- `milestone4_pipeline.py`
  Milestone 4 end-to-end integration pipeline:
  `load -> infer -> detect drift -> query human -> update model`.

- `milestone4_report.py`
  Milestone 4 final report artifact generator. Aggregates outputs from earlier phases and creates summary figures and diagrams.

- `demo_end_to_end.py`
  One-command demo wrapper for the Milestone 4 end-to-end pipeline and final report generation.

- `report_plots.py`
  Shared plotting functions used across robustness, monitoring, adaptation, and final reporting.

### `EEG data/`

Local dataset directory.

Expected structure:

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

- `sub_XX.json` files are the main source for training and evaluation.
- `P*.eeg` files are used by the raw audit workflow.

### `outputs/`

Stores Phase 1 audit outputs.

- `outputs/phase1_data_selection/`

### `results/`

Stores legacy baseline outputs from `app.py`.

- `results/baseline/torch_linear/...`
- `results/baseline/torch_mlp/...`

### `runs/`

Stores Milestone 2, 3, and 4 outputs.

- `runs/<training_run>/`
  Training runs from `train_pipeline.py`

- `runs/final_eval/<...>/`
  Milestone 3 evaluation artifacts

- `runs/milestone4/<...>/`
  Milestone 4 continual learning, HITL, active learning, system integration, demo, and final report artifacts

### Other important files

- `README.md`
  Main usage guide for the repository

- `MILESTONE4_SUBMISSION.md`
  Submission-oriented checklist and demo order for Milestone 4

- `pack_project.sh`
  Git-based pack script for creating a zip archive from committed `HEAD`

## 3. Important Run Directories

### Best verified training run

- `runs/20260314_095455_torch_mlp_expanded_baseline_mlp/`

This is the main trained model used by the Milestone 3 and Milestone 4 evaluation pipelines.

### Milestone 4 acceptance runs

- `runs/milestone4/20260406_152452_phase2_acceptance/`
- `runs/milestone4/20260406_153744_phase3_acceptance/`
- `runs/milestone4/20260406_155249_phase4_acceptance/`
- `runs/milestone4/20260406_161825_phase5_acceptance/`

### Milestone 4 final diagram/report directory

- `runs/milestone4/20260406_182729_phase6_diagrams/`

This directory contains the final summary figure set and the three workflow/system diagrams needed for the Milestone 4 report.

## 4. Where The Main Figures Are

### Phase 1 figures

Located in:

- `outputs/phase1_data_selection/`

Files:

- `plot_histograms.png`
- `plot_corr_heatmap.png`
- `plot_windows_per_participant.png`

### Milestone 2 training figures

Located in:

- `runs/20260314_095455_torch_mlp_expanded_baseline_mlp/`

Files:

- `learning_curves.png`
- `confusion_matrix.png`

### Milestone 3 figures

Failure analysis:

- `runs/final_eval/20260314_121454_report_failure/stress_test_summary.png`

Robustness and calibration:

- `runs/final_eval/20260314_122045_report_robustness/robustness_curve.png`
- `runs/final_eval/20260314_122045_report_robustness/reliability_diagram.png`
- `runs/final_eval/20260314_122045_report_robustness/confidence_histogram_clean.png`
- `runs/final_eval/20260314_122045_report_robustness/confidence_histogram_adv.png`
- `runs/final_eval/20260314_122045_report_robustness/efficiency_comparison.png`

Monitoring:

- `runs/final_eval/20260314_214803_report_monitoring/monitoring_dashboard.png`

Adaptation:

- `runs/final_eval/20260314_122638_report_adaptation/adaptation_before_after.png`
- `runs/final_eval/20260314_122638_report_adaptation/efficiency_comparison.png`

### Milestone 4 figures

Continual learning:

- `runs/milestone4/20260406_152452_phase2_acceptance/metric_trajectory.png`
- `runs/milestone4/20260406_152452_phase2_acceptance/update_effects.png`

HITL:

- `runs/milestone4/20260406_153744_phase3_acceptance/hitl_timeline.png`

Active learning:

- `runs/milestone4/20260406_155249_phase4_acceptance/active_learning_curve.png`

End-to-end system:

- `runs/milestone4/20260406_161825_phase5_acceptance/system_timeline.png`

Final report and diagrams:

- `runs/milestone4/20260406_182729_phase6_diagrams/system_comparison.png`
- `runs/milestone4/20260406_182729_phase6_diagrams/continual_learning_workflow.png`
- `runs/milestone4/20260406_182729_phase6_diagrams/active_learning_workflow.png`
- `runs/milestone4/20260406_182729_phase6_diagrams/system_architecture.png`

## 5. Where The Main Tables And Summaries Are

### Milestone 4 summary table

- `runs/milestone4/20260406_182729_phase6_diagrams/project_summary_table.csv`

### Final system summary

- `runs/milestone4/20260406_182729_phase6_diagrams/final_system_summary.json`

### Continual learning tables

- `runs/milestone4/20260406_152452_phase2_acceptance/update_history.csv`
- `runs/milestone4/20260406_152452_phase2_acceptance/update_decisions.csv`
- `runs/milestone4/20260406_152452_phase2_acceptance/model_versions.csv`
- `runs/milestone4/20260406_152452_phase2_acceptance/buffer_stats.csv`

### HITL tables

- `runs/milestone4/20260406_153744_phase3_acceptance/hitl_candidates.csv`
- `runs/milestone4/20260406_153744_phase3_acceptance/human_feedback_log.csv`
- `runs/milestone4/20260406_153744_phase3_acceptance/intervention_log.csv`

### Active learning tables

- `runs/milestone4/20260406_155249_phase4_acceptance/strategy_summary.csv`
- `runs/milestone4/20260406_155249_phase4_acceptance/cycle_metrics.csv`
- `runs/milestone4/20260406_155249_phase4_acceptance/query_log.csv`
- `runs/milestone4/20260406_155249_phase4_acceptance/labeling_efficiency.csv`

### End-to-end system tables

- `runs/milestone4/20260406_161825_phase5_acceptance/system_run_log.csv`
- `runs/milestone4/20260406_161825_phase5_acceptance/trigger_log.csv`
- `runs/milestone4/20260406_161825_phase5_acceptance/query_feedback_log.csv`
- `runs/milestone4/20260406_161825_phase5_acceptance/rollout_history.csv`
- `runs/milestone4/20260406_161825_phase5_acceptance/model_versions.csv`

## 6. Recommended Figure Set For The Milestone 4 Report

If you want the shortest useful figure list for the report, use these:

1. `continual_learning_workflow.png`
2. `metric_trajectory.png`
3. `hitl_timeline.png`
4. `active_learning_workflow.png`
5. `active_learning_curve.png`
6. `system_architecture.png`
7. `system_timeline.png`
8. `system_comparison.png`

And use this table with them:

- `project_summary_table.csv`

## 7. Recommended Demo Order

1. Show the trained baseline run under `runs/20260314_095455_torch_mlp_expanded_baseline_mlp/`
2. Show the continual learning artifacts under `runs/milestone4/20260406_152452_phase2_acceptance/`
3. Show HITL outputs under `runs/milestone4/20260406_153744_phase3_acceptance/`
4. Show active learning outputs under `runs/milestone4/20260406_155249_phase4_acceptance/`
5. Show the integrated system outputs under `runs/milestone4/20260406_161825_phase5_acceptance/`
6. Show the final report directory under `runs/milestone4/20260406_182729_phase6_diagrams/`

