from __future__ import annotations

import subprocess
import traceback
import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Iterable


@dataclass
class BatchSuccess:
    config_path: str
    seconds: float


@dataclass
class BatchFailure:
    config_path: str
    return_code: Any
    seconds: float
    output_tail: list[str] = field(default_factory=list)


class BatchSummaryLogger:
    """
    Collect batch run results and append a git-visible summary log.

    The class captures only the tail of child-process output for failed runs.
    This keeps the summary compact while preserving tracebacks and OOM messages.
    """

    def __init__(self, log_path: str | Path, error_tail_lines: int = 400):
        self.log_path = Path(log_path)
        self.error_tail_lines = error_tail_lines
        self.successes: list[BatchSuccess] = []
        self.failures: list[BatchFailure] = []

    @staticmethod
    def make_log_path(
        log_dir: str | Path,
        configs: Iterable[Any],
        *,
        prefix: str = "batch_summary",
        hash_extra: str = "",
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fingerprint_parts = [str(config) for config in configs]
        if hash_extra:
            fingerprint_parts.append(str(hash_extra))
        fingerprint = "\n".join(fingerprint_parts)
        batch_hash = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
        return Path(log_dir) / f"{prefix}-{timestamp}-{batch_hash}.log"

    @property
    def success_count(self) -> int:
        return len(self.successes)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    def make_output_tail(self) -> Deque[str]:
        return deque(maxlen=self.error_tail_lines)

    def subprocess_output_kwargs(self) -> dict:
        return {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }

    def capture_process_output(self, proc: subprocess.Popen, output_tail: Deque[str]) -> None:
        if proc.stdout is None:
            return

        # Text-mode pipes translate tqdm's carriage returns into separate
        # ``line`` values.  Printing those values normally produces hundreds of
        # stacked progress bars.  Detect tqdm render frames and redraw them on
        # the current terminal line instead.  Only the latest frame is retained
        # in the failure tail.
        tqdm_frame = re.compile(r"\s*\d+%\|.*\|\s*\d+/\d+")
        progress_active = False
        latest_progress = ""

        for line in proc.stdout:
            rendered = line.rstrip("\r\n")
            if tqdm_frame.search(rendered):
                print("\r" + rendered, end="", flush=True)
                latest_progress = rendered
                progress_active = True
                continue

            # ``leave=False`` may emit a whitespace-only frame to clear the bar.
            if progress_active and not rendered.strip():
                continue

            if progress_active:
                print()
                if latest_progress:
                    output_tail.append(latest_progress)
                progress_active = False
                latest_progress = ""

            print(line, end="")
            output_tail.append(rendered)

        if progress_active:
            print()
            if latest_progress:
                output_tail.append(latest_progress)

    def add_success(self, config_path: str, seconds: float) -> None:
        self.successes.append(BatchSuccess(config_path, seconds))

    def add_failure(
        self,
        config_path: str,
        return_code: Any,
        seconds: float,
        output_tail: Iterable[str] | None = None,
    ) -> None:
        self.failures.append(
            BatchFailure(
                config_path=config_path,
                return_code=return_code,
                seconds=seconds,
                output_tail=list(output_tail or []),
            )
        )

    def add_exception(self, config_path: str, seconds: float, return_code: Any = "exception") -> None:
        self.add_failure(config_path, return_code, seconds, traceback.format_exc().splitlines())

    def print_summary(self, total: int, style: str = "test") -> None:
        print("\n" + "=" * 60)
        if style == "entries":
            print(f"BATCH FINISHED -- {self.success_count} OK, {self.failure_count} FAILED")
            for failure in self.failures:
                print(
                    f"  FAIL: {failure.config_path}  "
                    f"(rc={failure.return_code}, elapsed={failure.seconds:.2f}s)"
                )
        else:
            print(f"[Batch][Summary] Total {total} | Success {self.success_count} | Failed {self.failure_count}")
            if self.successes:
                print("[Success examples] Top 3:")
                for success in self.successes[:3]:
                    print(f"  - {success.config_path} ({success.seconds:.2f}s)")
            if self.failures:
                print("[Failure list]")
                for failure in self.failures:
                    print(
                        f"  - {failure.config_path} | Return: {failure.return_code} "
                        f"| {failure.seconds:.2f}s"
                    )
        print("=" * 60)

    def write_summary(self, total: int) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"total={total} success={self.success_count} fail={self.failure_count}\n"
            )
            if self.successes:
                f.write("  successes:\n")
                for success in self.successes:
                    f.write(f"    - {success.config_path} ({success.seconds:.2f}s)\n")
            if self.failures:
                f.write("  failures:\n")
                for failure in self.failures:
                    f.write(
                        f"    - {failure.config_path} "
                        f"(return={failure.return_code}, elapsed={failure.seconds:.2f}s)\n"
                    )
                    if failure.output_tail:
                        f.write(f"      last_output_lines={len(failure.output_tail)}:\n")
                        for line in failure.output_tail:
                            f.write(f"        | {line}\n")
        return self.log_path

    def finalize(self, total: int, print_style: str = "test") -> None:
        self.print_summary(total, style=print_style)
        try:
            log_path = self.write_summary(total)
            print(f"[Batch] Results written to: {log_path}")
        except Exception as e:
            print(f"[Batch][WARN] Failed to write log: {e}")
