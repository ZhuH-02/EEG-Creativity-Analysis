# Milestone 4 Submission Guide

This repository now includes a full Milestone 4 path:

- continual learning
- human-in-the-loop review
- active learning
- end-to-end monitored rollout
- final reporting and demo scripts

## Recommended Command Order

Train or reuse the verified Milestone 2 run:

```powershell
python code/train_pipeline.py --model torch_mlp --tag expanded_baseline_mlp
```

Run the end-to-end Milestone 4 demo:

```powershell
python code/demo_end_to_end.py --model_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp --device cpu --with_report
```

Generate the final Phase 6 report directly:

```powershell
python code/milestone4_report.py `
  --training_run_dir runs/20260314_095455_torch_mlp_expanded_baseline_mlp `
  --phase2_dir runs/milestone4/20260406_152452_phase2_acceptance `
  --phase3_dir runs/milestone4/20260406_153744_phase3_acceptance `
  --phase4_dir runs/milestone4/20260406_155249_phase4_acceptance `
  --phase5_dir runs/milestone4/20260406_161825_phase5_acceptance
```

## Files To Show In The Demo

- `runs/<timestamp>_phase5.../system_timeline.png`
- `runs/<timestamp>_phase5.../rollout_history.csv`
- `runs/<timestamp>_phase5.../query_feedback_log.csv`
- `runs/<timestamp>_phase6.../project_summary_table.csv`
- `runs/<timestamp>_phase6.../system_comparison.png`
- `runs/<timestamp>_phase6.../submission_notes.md`

## Suggested Report Talking Points

1. Baseline subject-wise classifier performance.
2. Replay-based continual learning under drift.
3. HITL intervention triggers and review coverage.
4. Active learning strategy comparison.
5. Integrated system rollout, update traceability, and resource impact.

## Submission Checklist

- `README.md` updated with Milestone 4 workflow
- `code/milestone4_pipeline.py`
- `code/milestone4_report.py`
- `code/demo_end_to_end.py`
- `code/continual_learning.py`
- `code/hitl_active_learning.py`
- `code/active_learning_eval.py`
- latest `runs/milestone4/...` artifact directories
