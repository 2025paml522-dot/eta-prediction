"""
Compare a recent prediction window against the reference (training)
distribution using PSI / KS-test per feature and MAE degradation on
predictions with known ground truth (M5).

Simulates drift scenarios (e.g., festival/rush-hour surge) and raises a
retraining_trigger event when psi_threshold or MAE degradation thresholds
configured in config.yaml are exceeded.
"""
