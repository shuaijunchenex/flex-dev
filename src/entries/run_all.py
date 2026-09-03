import subprocess
import sys
import os
import time
import signal
from pathlib import Path
from typing import List

from flex.ml_utils import BatchSummaryLogger


def list_yaml_files(folder: str, target_path: str) -> List[str]:
    """
    List all .yaml and .yml files under a folder (non-recursive),
    and return paths as target_path + filename.
    """
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder} is not a valid directory")

    target = Path(target_path)
    files = []
    for p in sorted(list(folder_path.glob("*.yaml")) + list(folder_path.glob("*.yml"))):
        files.append(str(target / p.name))

    return files


# ── src/ directory (parent of entries/) ────────────────────────────────────
_BASE_DIR = Path(__file__).resolve().parent.parent


def run_all(configs, entry_module: str = "entries.standard", entry_func: str = "main"):
    child_code = (
        f"import sys; "
        f"import {entry_module}; "
        f"{entry_module}.{entry_func}(sys.argv[1])"
    )

    env = os.environ.copy()
    sep = ";" if os.name == "nt" else ":"
    env["PYTHONPATH"] = str(_BASE_DIR) + (sep + env.get("PYTHONPATH", ""))
    env["PYTHONUNBUFFERED"] = "1"

    summary = BatchSummaryLogger(
        BatchSummaryLogger.make_log_path(
            _BASE_DIR / "test",
            configs,
            hash_extra=f"{entry_module}.{entry_func}",
        )
    )

    stop_requested = False
    current_proc: subprocess.Popen | None = None

    def _terminate_current_process(proc: subprocess.Popen | None, timeout_sec: int = 5):
        if proc is None:
            return
        if proc.poll() is not None:
            return

        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass

    def _signal_handler(signum, _frame):
        nonlocal stop_requested, current_proc
        stop_requested = True
        print(f"\n[Batch][INTERRUPTED] Received signal {signum}. Stopping current job...")
        _terminate_current_process(current_proc)

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    total = len(configs)
    try:
        for idx, cfg in enumerate(configs, 1):
            if stop_requested:
                print("[Batch] Stop requested. Exiting batch loop.")
                break

            cfg_path = str(Path(cfg).resolve()) if os.path.isabs(cfg) else str((_BASE_DIR / "test" / cfg).resolve())
            print(f"\n[Batch] ({idx}/{total}) Running: {cfg_path}")
            cmd = [sys.executable, "-c", child_code, cfg_path]

            t0 = time.time()
            current_proc = None
            try:
                output_tail = summary.make_output_tail()
                current_proc = subprocess.Popen(
                    cmd,
                    **summary.subprocess_output_kwargs(),
                    env=env,
                    cwd=str(_BASE_DIR / "test"),
                    start_new_session=(os.name != "nt"),
                )
                summary.capture_process_output(current_proc, output_tail)
                ret = current_proc.wait()
                elapsed = time.time() - t0
                if ret == 0:
                    summary.add_success(cfg_path, elapsed)
                    print(f"[Batch] ({idx}/{total}) OK: {cfg_path} ({elapsed:.2f}s)")
                else:
                    summary.add_failure(cfg_path, ret, elapsed, output_tail)
                    print(f"[Batch] ({idx}/{total}) FAILED (rc={ret}): {cfg_path} ({elapsed:.2f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                summary.add_exception(cfg_path, elapsed, return_code=str(e))
                print(f"[Batch] ({idx}/{total}) EXCEPTION: {cfg_path} ({elapsed:.2f}s): {e}")
            finally:
                current_proc = None
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    summary.finalize(total, print_style="entries")
