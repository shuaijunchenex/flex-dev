from __future__ import annotations
import os
import json

import torch

from .csv_data_recorder import CsvDataRecorder
from .filename_maker import FileNameMaker


class TrainingLogger:
    """
    log train result to file
    """

    def __init__(self, config_dict: dict|None = None):
        """
        Init training logger
        Args:
            config_dict: dictionary, include "name", "path", "prefix"
        """
        d: dict = {}
        if config_dict is None:
            d = {}
        elif "training_logger" in config_dict and isinstance(config_dict["training_logger"], dict):
            # Wrapped shape: {"training_logger": {"name": ..., "path": ...}}
            d = config_dict["training_logger"]
        elif isinstance(config_dict, dict):
            # Direct shape: {"name": ..., "path": ...}
            # This is what most callers pass (e.g. self.config_dict["training_logger"]).
            d = config_dict

        self.name = d.get("name", "train")
        self.path: str = d.get("path", "./.training_results/")
        self.prefix: str = d.get("prefix", "")
        self.save_weights_enabled: bool = bool(d.get("save_weights", False))
        self.log_selected_clients: bool = bool(d.get("log_selected_clients", False))

        self.__file_names = None
        self.__logger = None
        self.__selection_log_stream = None   # combined: round + selected_clients + ewm weights
        self.__selection_log_metrics = None  # metric names from first ewm write
        self.__record_round = 0
        return

    def begin(self, header_config_dict: dict|None = None):
        """
        Write log begin
        """
        # Pass config content for content-based hashing
        config_str = json.dumps(header_config_dict, sort_keys=True, default=str) if header_config_dict else ""
        self.__file_names = (
            FileNameMaker
            .with_path(self.path)
            .with_prefix(self.prefix)
            .with_config_content(config_str)
            .make(self.name)
        )

        (path, _) = os.path.split(self.__file_names.fullname)
        os.makedirs(path, exist_ok=True)

        self.__logger = CsvDataRecorder(self.__file_names.fullname)
        self.__logger.begin(header_config_dict)
        self.__record_round = 0
        return

    def end(self):
        """
        Write log end
        """
        if hasattr(self, "_TrainingLogger__logger") and self.__logger is not None:
            self.__logger.end()
        if self.__selection_log_stream is not None:
            self.__selection_log_stream.close()
            self.__selection_log_stream = None
        self.__selection_log_metrics = None
        return

    def record(self, result_dict: dict):
        """
        Write record to CSV.

        The first call determines the CSV header columns; extra keys in later
        calls are silently dropped (backward-compatible with evaluator upgrades).
        """
        # ── Normalise round key ────────────────────────────────────────
        if "round" in result_dict:
            record = {"round": result_dict["round"],
                      **{k: v for k, v in result_dict.items() if k != "round"}}
        else:
            record = {"round": self.__record_round, **result_dict}
        self.__logger.record(record)
        self.__record_round += 1
        return self

    def record_evaluation(self, eval_metrics: dict, round_idx: int | None = None):
        """
        Log NLP evaluation results with round auto-tracking.

        Unlike ``record()``, this method ensures the round index is explicitly
        set for evaluation rows and auto-detects task-specific primary metrics.

        Args:
            eval_metrics: Dict returned by ``ModelEvaluator.evaluate()``.
            round_idx:    Optional round override; if None, uses internal counter.
        """
        r = round_idx if round_idx is not None else self.__record_round
        row = {"round": r, **eval_metrics}
        self.__logger.record(row)
        return self

    def save_weights(self, weight: dict, round_idx: int) -> str:
        """Save a model weight dict as a ``.pt`` file alongside the CSV log.

        The file is placed in the same directory as the CSV, with the same
        timestamp/hash prefix so it can be matched back to its run.
        Filename pattern::

            <path>/<prefix><name>-round<round_idx>-<timestamp>-<hash>.pt

        Args:
            weight:    The ``state_dict`` to persist (plain ``dict`` of tensors).
            round_idx: Current training round index, embedded in the filename.

        Returns:
            The absolute path of the saved file.
        """
        if self.__file_names is None:
            raise RuntimeError("TrainingLogger.begin() must be called before save_weights().")

        # Derive the weights filename from the CSV fullname by replacing the
        # extension and inserting the round index.
        csv_path = self.__file_names.fullname          # e.g. .../train-<ts>-<hash>.csv
        base = csv_path[:-4] if csv_path.endswith(".csv") else csv_path
        weights_path = f"{base}-round{round_idx:04d}.pt"

        os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
        torch.save({k: v.cpu() for k, v in weight.items()}, weights_path)
        return weights_path

    def save_weights_if_enabled(self, weight: dict, round_idx: int) -> str | None:
        """Call ``save_weights`` only when the ``save_weights`` YAML flag is ``true``.

        Args:
            weight:    The ``state_dict`` to persist.
            round_idx: Current training round index.

        Returns:
            The saved file path, or ``None`` if saving is disabled.
        """
        if not self.save_weights_enabled:
            return None
        return self.save_weights(weight, round_idx)

    def _open_selection_log(self, metric_names: list) -> None:
        """Lazily open the combined selection log and write the header."""
        if self.__file_names is None:
            return
        csv_path = self.__file_names.fullname
        log_path = (csv_path[:-4] if csv_path.endswith(".csv") else csv_path) + "_selection_log.csv"
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self.__selection_log_stream = open(log_path, "a", newline="", encoding="utf-8")
        self.__selection_log_metrics = metric_names
        header = "round,selected_clients," + ",".join(metric_names)
        self.__selection_log_stream.write(header + "\n")

    def record_selected_clients(self, round_idx: int, client_ids: list) -> None:
        """Buffer selected clients; actual write happens in record_ewm_weights.

        If ewm weights are never recorded, this is a no-op (by design — both
        pieces of information are written together in one row).
        """
        if not self.log_selected_clients:
            return
        # Store temporarily; flushed when record_ewm_weights is called
        self._pending_selected_clients = (round_idx, client_ids)

    def flush_selected_clients(self) -> None:
        """Write buffered selected clients immediately (no EWM weights).

        Use this when the selector does not produce EWM weights (e.g.
        high_loss, low_loss, random).  Writes a simple CSV row with
        ``round,selected_clients``.
        """
        if not self.log_selected_clients:
            return
        pending = getattr(self, "_pending_selected_clients", None)
        if pending is None:
            return
        round_idx, client_ids = pending
        self._pending_selected_clients = None

        if self.__selection_log_stream is None:
            self._open_selection_log([])

        clients_str = ";".join(str(c) for c in client_ids)
        self.__selection_log_stream.write(f"{round_idx},{clients_str}\n")
        self.__selection_log_stream.flush()

    def record_ewm_weights(self, round_idx: int, weights: dict) -> None:
        """Write one combined row: round, selected_clients, ewm_weight_1, ewm_weight_2, ...

        File pattern::

            <path>/<prefix><name>-<timestamp>-<hash>_selection_log.csv

        This method is a no-op when ``log_selected_clients`` is ``False``.
        """
        if not self.log_selected_clients:
            return
        if self.__file_names is None or not weights:
            return

        metric_names = list(weights.keys())

        if self.__selection_log_stream is None:
            self._open_selection_log(metric_names)

        # Get the buffered selected clients for this round (if any)
        pending = getattr(self, "_pending_selected_clients", None)
        if pending is not None and pending[0] == round_idx:
            clients_str = ";".join(str(c) for c in pending[1])
            self._pending_selected_clients = None
        else:
            clients_str = ""

        vals = ",".join(f"{weights[m]:.6f}" for m in metric_names)
        self.__selection_log_stream.write(f"{round_idx},{clients_str},{vals}\n")
        self.__selection_log_stream.flush()

    def __del__(self):
        self.end()
