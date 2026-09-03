from __future__ import annotations

import os
import sys

# ── Ensure CWD is the test/ directory for config path resolution ───────────
_ENTRIES_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_ENTRIES_DIR)
_TEST_DIR = os.path.join(_SRC_DIR, "test")
os.chdir(_TEST_DIR)
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, _SRC_DIR)

from flex.ml_utils import console
from entries.app.standard_entry import StandardSampleEntry
from flex.ml_utils.model_utils import ModelUtils

def main(config_path: str):
    g_app = StandardSampleEntry()
    g_app.load_app_config(config_path)
    device = ModelUtils.accelerator_device()
    g_app.run(device)

if __name__ == "__main__":
    main()
