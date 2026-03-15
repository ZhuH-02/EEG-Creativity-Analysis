# Failure Catalog

## covariate_shift
- Stage: input
- Description: Feature distribution changes relative to training windows can degrade learned decision boundaries.
- Mitigation: Monitor PSI and Wasserstein distance, retrain or recalibrate when drift persists.

## label_shift
- Stage: monitoring
- Description: Class prior changes can bias weighted metrics and confidence behavior.
- Mitigation: Track prediction class mix and recalibrate thresholds when priors move.

## concept_drift
- Stage: model
- Description: The relationship between EEG features and task phases may change over time or by cohort.
- Mitigation: Run periodic adaptation experiments and evaluate on fresh held-out participants.

## missing_critical_features
- Stage: input
- Description: Dropped or renamed features break checkpoint compatibility.
- Mitigation: Validate feature schema before inference and reject incompatible payloads.

## out_of_range_feature_values
- Stage: input
- Description: Extreme values indicate sensor issues, preprocessing bugs, or unseen operating conditions.
- Mitigation: Compare against training reference ranges and flag anomalous windows.

## sensor_channel_dropout
- Stage: signal
- Description: Missing channels alter the feature representation and can create brittle predictions.
- Mitigation: Stress-test with feature dropout and guard upstream sensor health.

## malformed_json_or_wrong_input_shape
- Stage: runtime
- Description: Bad payload shape or schema should fail fast instead of producing silent garbage outputs.
- Mitigation: Reject empty inputs, wrong dimensionality, and malformed feature tables.

## nan_inf_values
- Stage: runtime
- Description: NaN and Inf values can poison scaling, logits, and metrics.
- Mitigation: Block inference when non-finite values are detected.

## overconfident_wrong_predictions
- Stage: post_prediction
- Description: High-confidence errors are risky in downstream decision support.
- Mitigation: Track confidence calibration and support abstention thresholds.

## class_rarity_imbalance_degradation
- Stage: evaluation
- Description: Rare classes can collapse first under shift and imbalance.
- Mitigation: Report macro-F1, per-class metrics, and stress-test skewed class mixes.

## missing_checkpoint_or_schema_mismatch
- Stage: startup
- Description: Broken run folders or incompatible checkpoints should stop the evaluation cleanly.
- Mitigation: Validate required files before loading and surface clear errors.
