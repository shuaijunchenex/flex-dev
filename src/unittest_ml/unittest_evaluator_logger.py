"""
Unit test: ModelEvaluator NLP metrics + TrainingLogger CSV output.

Verifies:
  1. Perplexity is computed (exp(loss)).
  2. matthews_correlation / mcc are present and consistent.
  3. GLUE task → primary_metric auto-mapping (cola→mcc, sst2→accuracy).
  4. TrainingLogger.record() writes CSV correctly.
  5. CSV DictWriter extrasaction='ignore' tolerates changing dict keys.
  6. record_evaluation() embeds round index properly.

Uses a tiny dummy model and fake random data — no real dataset or GPU needed.

Run:
    cd unittest_ml
    python unittest_evaluator_logger.py
"""

from __future__ import annotations

import csv
import io
import math
import os
import tempfile
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------

from flex.model_trainer.model_evaluator import ModelEvaluator
from flex.ml_utils.training_logger import TrainingLogger
from flex.ml_utils.csv_data_recorder import CsvDataRecorder


# ======================================================================
# Tiny dummy model (3-class classifier, 4-dim input)
# ======================================================================

class DummyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("input_ids", next(iter(x.values())))
        return self.fc(x)


# ======================================================================
# Helpers
# ======================================================================

def _make_fake_glue_data(num_samples: int = 32, num_classes: int = 3):
    """Create a fake DataLoader mimicking a GLUE classification task."""
    torch.manual_seed(42)
    x = torch.randn(num_samples, 4)
    y = torch.randint(0, num_classes, (num_samples,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8)


# ======================================================================
# Test class
# ======================================================================

class TestEvaluatorNLP(unittest.TestCase):
    """Test ModelEvaluator NLP metrics with fake data."""

    def setUp(self):
        self.model = DummyClassifier()
        self.loader = _make_fake_glue_data(32, num_classes=3)

    # ------------------------------------------------------------------
    def test_all_metrics_present(self):
        """All classification + NLP metrics are in the result dict."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu")
        metrics = evaluator.evaluate()

        required = [
            "accuracy", "average_loss", "perplexity",
            "precision", "recall", "f1_score",
            "matthews_correlation", "mcc",
            "total_test_samples",
        ]
        for key in required:
            self.assertIn(key, metrics, f"Missing key: {key}")

    # ------------------------------------------------------------------
    def test_perplexity_formula(self):
        """Perplexity = exp(min(loss, 20))."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu")
        metrics = evaluator.evaluate()

        expected_ppl = math.exp(min(metrics["average_loss"], 20.0))
        self.assertAlmostEqual(metrics["perplexity"], expected_ppl, places=4)

    # ------------------------------------------------------------------
    def test_mcc_matthews_correlation_consistent(self):
        """mcc == matthews_correlation (alias)."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu")
        metrics = evaluator.evaluate()

        self.assertAlmostEqual(
            metrics["mcc"], metrics["matthews_correlation"], places=6,
            msg="mcc alias must equal matthews_correlation"
        )

    # ------------------------------------------------------------------
    def test_glue_task_cola_primary_mcc(self):
        """CoLA task → primary_metric = matthews_correlation."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu",
                                   task_name="cola")
        metrics = evaluator.evaluate()

        self.assertEqual(metrics.get("primary_metric"), "matthews_correlation")
        self.assertIn("primary_score", metrics)
        self.assertAlmostEqual(
            metrics["primary_score"], metrics["matthews_correlation"], places=6
        )

    # ------------------------------------------------------------------
    def test_glue_task_sst2_primary_accuracy(self):
        """SST-2 task → primary_metric = accuracy."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu",
                                   task_name="sst2")
        metrics = evaluator.evaluate()

        self.assertEqual(metrics.get("primary_metric"), "accuracy")
        self.assertAlmostEqual(
            metrics["primary_score"], metrics["accuracy"], places=6
        )

    # ------------------------------------------------------------------
    def test_unknown_task_no_primary(self):
        """Unknown task → no primary_metric/primary_score keys."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu",
                                   task_name="unknown_task_xyz")
        metrics = evaluator.evaluate()

        self.assertNotIn("primary_metric", metrics)
        self.assertNotIn("primary_score", metrics)

    # ------------------------------------------------------------------
    def test_convenience_accessors(self):
        """get_accuracy / get_loss / get_perplexity / get_mcc / get_primary_score."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu",
                                   task_name="cola")
        evaluator.evaluate()

        self.assertIsNotNone(evaluator.get_accuracy())
        self.assertIsNotNone(evaluator.get_loss())
        self.assertIsNotNone(evaluator.get_perplexity())
        self.assertIsNotNone(evaluator.get_matthews_correlation())
        self.assertIsNotNone(evaluator.get_primary_score())

    # ------------------------------------------------------------------
    def test_pipeline_test_quick(self):
        """pipeline_test=True evaluates on a single sample only."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu")
        metrics = evaluator.evaluate(pipeline_test=True)
        self.assertEqual(metrics["total_test_samples"], 1)

    # ------------------------------------------------------------------
    def test_print_results_no_error(self):
        """print_results() should not raise when metrics exist."""
        evaluator = ModelEvaluator(self.model, self.loader, device="cpu",
                                   task_name="cola")
        evaluator.evaluate()
        # Should not raise
        evaluator.print_results()


# ======================================================================
# Test class: Logger + CSV DictWriter compatibility
# ======================================================================

class TestLoggerCSV(unittest.TestCase):
    """Test TrainingLogger and CsvDataRecorder with evolving dict keys."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="unittest_logger_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_csv_writer_extrasaction_ignore(self):
        """DictWriter with extrasaction='ignore' tolerates extra keys."""
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream, fieldnames=["round", "accuracy", "loss"],
            extrasaction="ignore", restval="",
        )
        writer.writeheader()

        # First row: only known keys
        writer.writerow({"round": 0, "accuracy": 0.8, "loss": 0.5})

        # Second row: includes a new key not in the header
        writer.writerow({
            "round": 1, "accuracy": 0.9, "loss": 0.4,
            "perplexity": 1.5,  # <-- extra key, should be ignored
        })

        # Should NOT raise ValueError
        output = stream.getvalue()
        self.assertIn("round,accuracy,loss", output)
        self.assertIn("0,0.8,0.5", output)
        self.assertIn("1,0.9,0.4", output)
        self.assertNotIn("perplexity", output.splitlines()[0])  # not in header

    # ------------------------------------------------------------------
    def test_record_evaluation_writes_csv(self):
        """record_evaluation() writes round + eval metrics to CSV."""
        logger = TrainingLogger({
            "name": "test_eval",
            "path": self.tmpdir,
            "prefix": "ut_",
        })
        logger.begin({"experiment": "unittest"})

        logger.record_evaluation({
            "accuracy": 0.85, "average_loss": 0.42,
            "perplexity": math.exp(0.42),
            "matthews_correlation": 0.65,
            "primary_metric": "matthews_correlation",
            "primary_score": 0.65,
        }, round_idx=0)

        logger.record_evaluation({
            "accuracy": 0.88, "average_loss": 0.38,
            "perplexity": math.exp(0.38),
            "matthews_correlation": 0.70,
            "primary_metric": "matthews_correlation",
            "primary_score": 0.70,
        }, round_idx=1)

        logger.end()

        # Read CSV back and verify
        csv_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".csv")]
        self.assertEqual(len(csv_files), 1, "Expected exactly 1 CSV file")

        csv_path = os.path.join(self.tmpdir, csv_files[0])
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Header must include round + NLP metrics
        self.assertIn("round", content)
        self.assertIn("perplexity", content)
        self.assertIn("matthews_correlation", content)
        self.assertIn("primary_metric", content)
        self.assertIn("primary_score", content)

        # Both rounds must be present
        self.assertIn("0,", content.splitlines()[-2])  # round 0
        self.assertIn("1,", content.splitlines()[-1])  # round 1

    # ------------------------------------------------------------------
    def test_changing_keys_between_rounds(self):
        """When evaluator returns different keys in later rounds, no crash."""
        logger = TrainingLogger({
            "name": "test_changing_keys",
            "path": self.tmpdir,
            "prefix": "ut_",
        })
        logger.begin({"experiment": "unittest"})

        # Round 0: old-style metrics (5 keys)
        logger.record_evaluation({
            "accuracy": 0.8, "average_loss": 0.5,
            "precision": 0.7, "recall": 0.6, "f1_score": 0.65,
        }, round_idx=0)

        # Round 1: new-style metrics (8 keys — perplexity & mcc added)
        logger.record_evaluation({
            "accuracy": 0.85, "average_loss": 0.42,
            "precision": 0.75, "recall": 0.68, "f1_score": 0.71,
            "perplexity": 1.52,
            "matthews_correlation": 0.65,
            "primary_score": 0.65,
        }, round_idx=1)

        logger.end()

        # Must not have raised ValueError
        csv_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".csv")]
        csv_path = os.path.join(self.tmpdir, csv_files[0])
        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Both rounds recorded
        self.assertIn("0,", content)
        self.assertIn("1,", content)

    # ------------------------------------------------------------------
    def test_record_basic_round_auto_increment(self):
        """record() auto-increments round when not provided in dict."""
        logger = TrainingLogger({
            "name": "test_auto_round",
            "path": self.tmpdir,
            "prefix": "ut_",
        })
        logger.begin()

        logger.record({"accuracy": 0.5})
        logger.record({"accuracy": 0.6})
        logger.end()

        csv_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".csv")]
        csv_path = os.path.join(self.tmpdir, csv_files[0])
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Find data rows (skip config header)
        data_lines = [l for l in lines if l.strip() and not l.startswith("Config")]
        self.assertIn("0,0.5", data_lines[1] if len(data_lines) > 1 else "")
        self.assertIn("1,0.6", data_lines[2] if len(data_lines) > 2 else "")


if __name__ == "__main__":
    unittest.main()
