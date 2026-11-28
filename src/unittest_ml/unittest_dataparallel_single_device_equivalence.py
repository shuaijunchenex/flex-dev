from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

# Init startup path, change current path to test py file folder
# -----------------------------------------------------------------
import os
from startup_init import startup_init_path

startup_init_path(os.path.dirname(os.path.abspath(__file__)))
# -----------------------------------------------------------------

from flex.ml_utils.model_utils import ModelUtils
from flex.model_trainer.model_trainer_args import ModelTrainerArgs
from flex.model_trainer.trainer._model_trainer_standard import ModelTrainer_Standard


class TestDataParallelSingleDeviceEquivalence(unittest.TestCase):
    """Regression tests: single-device behavior must match non-wrap behavior."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    @staticmethod
    def _build_model() -> nn.Module:
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    @staticmethod
    def _assert_state_dict_equal(tc: unittest.TestCase, a: dict, b: dict, atol: float = 1e-7):
        tc.assertEqual(list(a.keys()), list(b.keys()))
        for k in a.keys():
            tc.assertTrue(torch.allclose(a[k], b[k], atol=atol), msg=f"Mismatch on key: {k}")

    def test_wrap_cpu_forward_is_equivalent_to_no_wrap(self):
        model_raw = self._build_model().to(self.device)
        model_wrapped = self._build_model().to(self.device)
        model_wrapped.load_state_dict(copy.deepcopy(model_raw.state_dict()))

        wrapped = ModelUtils.wrap_data_parallel(model_wrapped, self.device)
        self.assertFalse(isinstance(wrapped, nn.DataParallel))

        x = torch.randn(6, 8, device=self.device)
        y_raw = model_raw(x)
        y_wrapped = wrapped(x)

        self.assertTrue(torch.allclose(y_raw, y_wrapped, atol=1e-7))

    def test_wrap_cpu_one_step_update_is_equivalent_to_no_wrap(self):
        model_raw = self._build_model().to(self.device)
        model_wrapped = self._build_model().to(self.device)
        model_wrapped.load_state_dict(copy.deepcopy(model_raw.state_dict()))

        wrapped = ModelUtils.wrap_data_parallel(model_wrapped, self.device)
        self.assertFalse(isinstance(wrapped, nn.DataParallel))

        opt_raw = optim.SGD(model_raw.parameters(), lr=0.01)
        opt_wrapped = optim.SGD(wrapped.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(10, 8, device=self.device)
        y = torch.randint(0, 4, (10,), device=self.device)

        opt_raw.zero_grad()
        loss_raw = criterion(model_raw(x), y)
        loss_raw.backward()
        opt_raw.step()

        opt_wrapped.zero_grad()
        loss_wrapped = criterion(wrapped(x), y)
        loss_wrapped.backward()
        opt_wrapped.step()

        self.assertAlmostEqual(loss_raw.item(), loss_wrapped.item(), places=7)
        self._assert_state_dict_equal(self, model_raw.state_dict(), ModelUtils.unwrap_model(wrapped).state_dict())

    def test_unwrap_state_dict_keys_match_legacy_format(self):
        model = self._build_model().to(self.device)
        wrapped = ModelUtils.wrap_data_parallel(model, self.device)

        legacy_keys = list(model.state_dict().keys())
        new_keys = list(ModelUtils.unwrap_model(wrapped).state_dict().keys())

        self.assertEqual(legacy_keys, new_keys)
        self.assertFalse(any(k.startswith("module.") for k in new_keys))

    def test_standard_trainer_cpu_set_model_stays_single_device_compatible(self):
        base_model = self._build_model().to(self.device)
        args = ModelTrainerArgs()
        args.model = base_model
        args.optimizer = optim.SGD(base_model.parameters(), lr=0.01)
        args.device = "cpu"

        trainer = ModelTrainer_Standard(args)
        self.assertFalse(isinstance(trainer.trainer_args.model, nn.DataParallel))

        new_model = self._build_model().to(self.device)
        trainer.set_model(new_model)

        self.assertFalse(isinstance(trainer.trainer_args.model, nn.DataParallel))
        self.assertIs(trainer.trainer_args.model, trainer.model)
        self.assertEqual(
            list(trainer.model.state_dict().keys()),
            list(ModelUtils.unwrap_model(trainer.model).state_dict().keys()),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
