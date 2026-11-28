import io
import sys
from collections import deque
from pathlib import Path
from unittest.mock import patch

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from flex.ml_utils.batch_summary import BatchSummaryLogger
from flex.ml_utils.tqdm_utils import TqdmWrapper


def test_wrapper_enforces_single_terminal_slot() -> None:
    wrapper = TqdmWrapper(leave=True, position=7, ascii=False)

    with patch("flex.ml_utils.tqdm_utils._auto_tqdm") as mocked_tqdm:
        wrapper(
            range(2),
            leave=True,
            position=3,
            ascii=False,
        )

    kwargs = mocked_tqdm.call_args.kwargs
    assert kwargs["leave"] is False
    assert kwargs["position"] == 0
    assert kwargs["ascii"] is True


def test_batch_capture_retains_only_latest_tqdm_frame() -> None:
    class Process:
        stdout = io.StringIO(
            "Loading:   0%|          | 0/10 [00:00<?, ?batch/s]\n"
            "Loading:  50%|#####     | 5/10 [00:01<00:01, 5batch/s]\n"
            "Loading: 100%|##########| 10/10 [00:02<00:00, 5batch/s]\n"
        )

    output_tail = deque(maxlen=20)
    output = io.StringIO()
    logger = BatchSummaryLogger("unused.log")

    with patch("sys.stdout", output):
        logger.capture_process_output(Process(), output_tail)

    assert output.getvalue().count("\r") == 3
    assert output.getvalue().count("\n") == 1
    assert list(output_tail) == [
        "Loading: 100%|##########| 10/10 [00:02<00:00, 5batch/s]"
    ]

if __name__ == "__main__":
    test_wrapper_enforces_single_terminal_slot()
    test_batch_capture_retains_only_latest_tqdm_frame()
    print("All tests passed.")