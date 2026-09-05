"""Evaluation protocol and scoring.

``splitting``        train/test division and target-distribution reporting
``metrics``          task-appropriate metrics, their directions and the primary metric
``cross_validation`` scoring a model over training folds
``diagnostics``      signals worth a second look in a finished run

``diagnostics`` computes no metrics of its own. It reads the numbers the other
modules already produced and names the situations a reader should look at
twice — a gap between cross-validated and held-out scores, folds that
disagreed, a tiny test split. They are signals, never verdicts.
"""
