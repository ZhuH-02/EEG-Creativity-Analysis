# Milestone 4 Submission Notes

## Source Artifacts
- Training baseline: `runs\20260314_095455_torch_mlp_expanded_baseline_mlp`
- Phase 2 continual learning: `runs\milestone4\20260406_152452_phase2_acceptance`
- Phase 3 HITL: `runs\milestone4\20260406_153744_phase3_acceptance`
- Phase 4 active learning: `runs\milestone4\20260406_155249_phase4_acceptance`
- Phase 5 end-to-end system: `runs\milestone4\20260406_161825_phase5_acceptance`

## Key Results
- Best active learning strategy: `uncertainty`
- Phase 5 final clean macro-F1: `0.3330`
- Phase 5 final drifted macro-F1: `0.3727`
- Phase 5 total queries: `192`
- Phase 5 updates completed: `3`

## Suggested Demo Order
1. Show the baseline training run metrics and efficiency.
2. Show Phase 2 replay-based continual learning outputs.
3. Show Phase 3 HITL candidate and feedback logs.
4. Show Phase 4 active learning curve and strategy summary.
5. Show Phase 5 end-to-end timeline and rollout history.

## Recommended Files to Mention in the Report
- `project_summary_table.csv`
- `final_system_summary.json`
- `system_comparison.png`
- `continual_learning_workflow.png`
- `active_learning_workflow.png`
- `system_architecture.png`
- `submission_notes.md`
