import contextlib
import math
from typing import Optional

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import torch
import torch.nn as nn

from ..ml_utils import console
from ..ml_utils.tqdm_utils import pbar
from ..ml_utils.training_utils import TrainingUtils


# ── GLUE task → primary metric mapping ────────────────────────────────────
_GLUE_PRIMARY_METRIC: dict[str, str] = {
    "cola":  "matthews_correlation",
    "sst2":  "accuracy",
    "mrpc":  "f1_score",       # also reports accuracy
    "qqp":   "f1_score",       # also reports accuracy
    "mnli":  "accuracy",
    "qnli":  "accuracy",
    "rte":   "accuracy",
    "wnli":  "accuracy",
    "stsb":  "pearson",        # regression task
}

class ModelEvaluator:
    """
    A stateful evaluator for PyTorch models.
    Initialized with model, validation dataloader, and device.
    """

    def __init__(self, model, val_loader, criterion=None, device="cpu",
                 task_name: Optional[str] = None,
                 legacy_eval_rng_compat: bool = False):
        """
        :param model:       PyTorch model to evaluate
        :param val_loader:  DataLoader with validation or test data
        :param criterion:   Loss function (e.g., CrossEntropyLoss). If None, uses CrossEntropyLoss
        :param device:      Computation device ('cpu' or 'cuda')
        :param task_name:   Optional GLUE task name (e.g. 'cola', 'sst2') used to
                            auto-select the primary metric for this task.
        """

        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.task_name = (task_name or "").lower()
        self.legacy_eval_rng_compat = legacy_eval_rng_compat

        # Default to CrossEntropyLoss if no criterion provided
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.latest_metrics = {}

    def change_model(self, model, weight=None):
        if weight is not None:
            model.load_state_dict(weight, strict=True)
        self.model = model
        # Keep on CPU — callers should move to device before evaluate()
        return self

    def update_model(self, weight):
        self.model.load_state_dict(weight, strict=True)

    def evaluate(self, average="macro", pipeline_test: bool = False):
        if self.legacy_eval_rng_compat:
            return self._evaluate_legacy(average=average, pipeline_test=pipeline_test)

        self.model.eval()
        self.model = self.model.to(self.device)
        all_preds, all_labels = [], []
        prediction_outputs = None
        invalid_label_samples = 0
        total_loss, total_samples = 0.0, 0
        total_correct = 0

        if pipeline_test:
            console.info("[ModelEvaluator] pipeline_test mode 鈥?evaluating on ONE sample only.")

        amp_dtype = TrainingUtils.resolve_amp_dtype(self.device)
        dev_type = self.device.type if hasattr(self.device, "type") else str(self.device).split(":")[0]
        amp_ctx = torch.autocast(device_type=dev_type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()

        val_loader = getattr(self.val_loader, "test_data_loader", self.val_loader)
        loop = pbar(val_loader, desc="Evaluating", leave=False, ncols=100)
        with torch.inference_mode(), amp_ctx:
            for inputs, labels in loop:
                if hasattr(inputs, "to"):
                    inputs = TrainingUtils.to_device(inputs, self.device)
                elif isinstance(inputs, dict):
                    inputs = {k: (TrainingUtils.to_device(v, self.device) if hasattr(v, "to") else v)
                              for k, v in inputs.items()}
                labels = TrainingUtils.to_device(labels, self.device).long()

                if pipeline_test:
                    if isinstance(inputs, dict) or (hasattr(inputs, 'items') and hasattr(inputs, 'keys')):
                        inputs = type(inputs)(
                            {k: (v[:1] if torch.is_tensor(v) else v)
                             for k, v in inputs.items()}
                        ) if type(inputs) is not dict else \
                            {k: (v[:1] if torch.is_tensor(v) else v)
                             for k, v in inputs.items()}
                    elif torch.is_tensor(inputs):
                        inputs = inputs[:1]
                    labels = labels[:1]

                outputs = self.model(inputs)

                num_classes = outputs.shape[1] if outputs.dim() > 1 else 1
                predicted = outputs.argmax(dim=1)
                valid_mask = (labels >= 0) & (labels < num_classes)
                invalid_count = int((~valid_mask).sum().item())
                if invalid_count > 0:
                    if invalid_label_samples == 0:
                        min_label = labels.min().item()
                        max_label = labels.max().item()
                        console.warn(
                            "Label values out of range "
                            f"[{min_label}, {max_label}] for num_classes={num_classes}. "
                            "Invalid labels will not be changed and will be excluded "
                            "from loss and metrics. If all labels are invalid (as in a "
                            "GLUE test split), predictions will be returned without "
                            "supervised metrics."
                        )
                    if prediction_outputs is None:
                        prediction_outputs = list(all_preds)
                    invalid_label_samples += invalid_count

                if prediction_outputs is not None:
                    prediction_outputs.extend(predicted.cpu().tolist())

                valid_count = int(valid_mask.sum().item())
                if valid_count > 0:
                    valid_outputs = outputs[valid_mask]
                    valid_labels = labels[valid_mask]
                    valid_predictions = predicted[valid_mask]
                    loss = self.criterion(valid_outputs, valid_labels)

                    total_loss += loss.item() * valid_count
                    total_samples += valid_count
                    all_preds.extend(valid_predictions.cpu().tolist())
                    all_labels.extend(valid_labels.cpu().tolist())
                    total_correct += int(
                        (valid_predictions == valid_labels).sum().item()
                    )

                    running_acc = total_correct / max(total_samples, 1)
                    loop.set_postfix(
                        acc=f"{running_acc:.4f}", loss=f"{loss.item():.4f}"
                    )
                else:
                    loop.set_postfix(status="prediction_only")

                del inputs, labels, outputs

                if pipeline_test:
                    break

        avg_loss = total_loss / total_samples if total_samples > 0 else None
        self.latest_metrics = self._compute_metrics(
            all_labels,
            all_preds,
            avg_loss,
            total_samples,
            average,
            predictions=prediction_outputs,
            invalid_label_samples=invalid_label_samples,
        )

        self.model = self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        del all_preds, all_labels
        return self.latest_metrics

    def _evaluate_legacy(self, average="macro", pipeline_test: bool = False):
        """
        Evaluate the model on the validation dataset.

        :param average: Averaging strategy for precision/recall/F1 (default: 'macro').
        :param pipeline_test: When True, run evaluation on **one sample only** to
                              verify the full pipeline quickly without waiting for the
                              entire validation set.
        """
        self.model.eval()
        self.model = self.model.to(self.device)
        all_preds, all_labels = [], []
        prediction_outputs = None
        invalid_label_samples = 0
        total_loss, total_samples = 0.0, 0

        if pipeline_test:
            console.info("[ModelEvaluator] pipeline_test mode — evaluating on ONE sample only.")

        with torch.inference_mode():
            for inputs, labels in getattr(self.val_loader, "test_data_loader", self.val_loader):
                # Move inputs/labels to device; handle HF BatchEncoding/dict
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.device, non_blocking=True)
                elif isinstance(inputs, dict):
                    inputs = {k: (v.to(self.device, non_blocking=True) if hasattr(v, "to") else v)
                              for k, v in inputs.items()}
                labels = labels.to(self.device).long()

                # ---- pipeline_test: truncate to a single sample ----
                if pipeline_test:
                    # BatchEncoding (transformers ≥4.45) is not a dict subclass;
                    # use duck-typing: check for .items() method.
                    if isinstance(inputs, dict) or (hasattr(inputs, 'items') and hasattr(inputs, 'keys')):
                        # dict / BatchEncoding / mapping → truncate every tensor value
                        inputs = type(inputs)(  # preserve original type where possible
                            {k: (v[:1] if torch.is_tensor(v) else v)
                             for k, v in inputs.items()}
                        ) if type(inputs) is not dict else \
                            {k: (v[:1] if torch.is_tensor(v) else v)
                             for k, v in inputs.items()}
                    elif torch.is_tensor(inputs):
                        inputs = inputs[:1]
                    labels = labels[:1]

                outputs = self.model(inputs)

                num_classes = outputs.shape[1] if outputs.dim() > 1 else 1
                predicted = outputs.argmax(dim=1)
                valid_mask = (labels >= 0) & (labels < num_classes)
                invalid_count = int((~valid_mask).sum().item())
                if invalid_count > 0:
                    if invalid_label_samples == 0:
                        min_label = labels.min().item()
                        max_label = labels.max().item()
                        console.warn(
                            "Label values out of range "
                            f"[{min_label}, {max_label}] for num_classes={num_classes}. "
                            "Invalid labels will not be changed and will be excluded "
                            "from loss and metrics. If all labels are invalid (as in a "
                            "GLUE test split), predictions will be returned without "
                            "supervised metrics."
                        )
                    if prediction_outputs is None:
                        prediction_outputs = list(all_preds)
                    invalid_label_samples += invalid_count

                if prediction_outputs is not None:
                    prediction_outputs.extend(predicted.cpu().tolist())

                valid_count = int(valid_mask.sum().item())
                if valid_count > 0:
                    valid_outputs = outputs[valid_mask]
                    valid_labels = labels[valid_mask]
                    valid_predictions = predicted[valid_mask]
                    loss = self.criterion(valid_outputs, valid_labels)

                    total_loss += loss.item() * valid_count
                    total_samples += valid_count
                    all_preds.extend(valid_predictions.cpu().tolist())
                    all_labels.extend(valid_labels.cpu().tolist())

                # pipeline_test: one sample is enough, stop immediately
                if pipeline_test:
                    break

        avg_loss = total_loss / total_samples if total_samples > 0 else None
        self.latest_metrics = self._compute_metrics(
            all_labels,
            all_preds,
            avg_loss,
            total_samples,
            average,
            predictions=prediction_outputs,
            invalid_label_samples=invalid_label_samples,
        )

        # ── Move model back to CPU to free GPU memory ───────────────────
        self.model = self.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.latest_metrics

    def _compute_metrics(
        self,
        all_labels,
        all_preds,
        avg_loss,
        total_samples,
        average,
        predictions=None,
        invalid_label_samples=0,
    ):
        if total_samples == 0:
            metrics = {
                "accuracy": None,
                "average_loss": None,
                "perplexity": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "matthews_correlation": None,
                "mcc": None,
                "total_test_samples": 0,
                "total_prediction_samples": len(predictions or []),
                "invalid_label_samples": invalid_label_samples,
                "evaluation_status": "prediction_only",
                "predictions": predictions or [],
            }
            if self.task_name in _GLUE_PRIMARY_METRIC:
                metrics["primary_metric"] = _GLUE_PRIMARY_METRIC[self.task_name]
                metrics["primary_score"] = None
            return metrics

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "average_loss": avg_loss,
            "perplexity": math.exp(min(avg_loss, 20.0)),  # clamp to avoid overflow
            "precision": precision_score(all_labels, all_preds, average=average, zero_division=0),
            "recall": recall_score(all_labels, all_preds, average=average, zero_division=0),
            "f1_score": f1_score(all_labels, all_preds, average=average, zero_division=0),
            "matthews_correlation": matthews_corrcoef(all_labels, all_preds),
            "mcc": matthews_corrcoef(all_labels, all_preds),  # alias for backward compat
            "total_test_samples": total_samples,
            "invalid_label_samples": invalid_label_samples,
            "evaluation_status": (
                "partial_labels" if invalid_label_samples > 0 else "evaluated"
            ),
        }
        if invalid_label_samples > 0:
            metrics["total_prediction_samples"] = len(predictions or [])
            metrics["predictions"] = predictions or []

        if self.task_name in _GLUE_PRIMARY_METRIC:
            primary_key = _GLUE_PRIMARY_METRIC[self.task_name]
            metrics["primary_metric"] = primary_key
            metrics["primary_score"] = metrics.get(primary_key, 0.0)

        return metrics
        
    def print_results(self):
        """
        Pretty-print the latest evaluation metrics.
        Should be called after evaluate().
        """

        if not self.latest_metrics:
            console.error("No evaluation metrics available. run .evaluate() first.")
            return

        m = self.latest_metrics
        console.info("Evaluation Summary:")

        if m.get("evaluation_status") == "prediction_only":
            console.warn(
                "  - No valid labels were available; supervised metrics were not "
                "computed."
            )
            console.info(
                f"  - Predictions : {m.get('total_prediction_samples', 0)}"
            )
            console.info(
                f"  - Invalid labels: {m.get('invalid_label_samples', 0)}"
            )
            return

        # ── Task context ───────────────────────────────────────────────
        if self.task_name:
            primary = m.get("primary_metric", "—")
            score   = m.get("primary_score", 0.0)
            if isinstance(score, float):
                console.info(f"  * Task ({self.task_name}) → {primary}: {score:.4f}")

        console.info(f"  - Loss        : {m['average_loss']:.4f}")
        console.info(f"  - Perplexity  : {m.get('perplexity', 0.0):.4f}")
        console.info(f"  - Accuracy    : {m['accuracy'] * 100:.2f}%")
        console.info(f"  - Precision   : {m['precision']:.4f}")
        console.info(f"  - Recall      : {m['recall']:.4f}")
        console.info(f"  - F1-Score    : {m['f1_score']:.4f}")
        console.info(f"  - MCC         : {m.get('matthews_correlation', 0.0):.4f}")
        console.info(f"  - Samples     : {m['total_test_samples']}")
        return

    # ── Convenience accessors ──────────────────────────────────────────────
    def get_accuracy(self):
        """Quick access to accuracy."""
        return self.latest_metrics.get('accuracy', None)

    def get_loss(self):
        """Quick access to average loss."""
        return self.latest_metrics.get('average_loss', None)

    def get_perplexity(self) -> Optional[float]:
        """Quick access to perplexity (NLP)."""
        return self.latest_metrics.get('perplexity', None)

    def get_matthews_correlation(self) -> Optional[float]:
        """Quick access to Matthews Correlation Coefficient (e.g. CoLA)."""
        return self.latest_metrics.get('matthews_correlation', None)

    def get_primary_score(self) -> Optional[float]:
        """Quick access to the task-specific primary metric score."""
        return self.latest_metrics.get('primary_score', None)

