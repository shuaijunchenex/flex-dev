"""ModelEvaluator variant that reports per-batch supervision to a callback."""
from __future__ import annotations

from flex.model_trainer.model_evaluator import ModelEvaluator


class ModelEvaluatorBatchObserver(ModelEvaluator):
    """Forward ``(labels, predictions, valid_mask, per-sample loss)`` for every batch.

    Used by the SFL evaluation path to accumulate confusion matrices and per-domain
    counters from the same pass that computes the metrics.
    """

    def __init__(self, *args, observer, **kwargs):
        super().__init__(*args, **kwargs)
        self._observer = observer

    def _on_evaluate_batch(self, labels, predictions, valid_mask, loss):
        if loss.ndim != 1:
            raise ValueError("batch observer requires a criterion returning one loss per valid sample")
        self._observer(labels, predictions, valid_mask, loss)
