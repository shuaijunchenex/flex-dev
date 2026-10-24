from __future__ import annotations

import os
import sys

_ENTRIES_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_ENTRIES_DIR)
_TEST_DIR = os.path.join(_SRC_DIR, "test")
os.chdir(_TEST_DIR)
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, _SRC_DIR)

from flex.ml_utils import console
from entries.app.lora_entry import LoRAAppEntry
from flex.ml_utils.model_utils import ModelUtils
from flex.ml_utils.training_utils import TrainingUtils

g_app = LoRAAppEntry()

def main(config_path: str):
    g_app.load_app_config(config_path)
    device = ModelUtils.accelerator_device()
    g_app.run(device)

if __name__ == "__main__":
    TrainingUtils.set_seed_all(42)
    console.set_log_level("all")
    console.set_debug(True)
    console.set_console_logger(log_path="./log/", log_name="console_trace")
    console.set_exception_logger(log_path="./log/", log_name="exception_trace")
    console.set_debug_logger(log_path="./log/", log_name="debug_trace")
    console.enable_console_log(True)
    console.enable_exception_log(True)
    console.enable_debug_log(True)
    console.out("RBLA program")
    console.out("======================= PROGRAM BEGIN ==========================")
    main()
    console.out("\n======================= PROGRAM END ============================")
