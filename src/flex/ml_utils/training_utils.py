import random
import contextlib
from typing import Any, Dict, Optional

import numpy as np
import torch

from ..ml_utils import console


# ── Optimizations that may affect bit-exact reproducibility ────────────────
_BIT_EXACT_SENSITIVE_OPTS = frozenset({
    "cudnn_benchmark",
    "torch_compile",
})


class TrainingUtils:
    # ------------------------------------------------------------------
    # Per-process train-optimization config (set once via apply_train_optimization)
    # ------------------------------------------------------------------
    _train_opt_config: Dict[str, bool] = {}

    @classmethod
    def apply_train_optimization(cls, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Read ``general.train_optimization`` from *config_dict* and apply
        infrastructure-level settings (cuDNN benchmark).

        Call once at app startup after config loading.  If *config_dict* is
        None or missing the section, all optimizations default to ``False``
        (safe, bit-exact mode).
        """
        cfg: Dict[str, Any] = {}
        if config_dict is not None:
            cfg = config_dict.get("general", {}).get("train_optimization", {})

        cls._train_opt_config = {
            "cudnn_benchmark":        bool(cfg.get("cudnn_benchmark", False)),
            "pin_memory":             bool(cfg.get("pin_memory", False)),
            "fused_optimizer":        bool(cfg.get("fused_optimizer", False)),
            "torch_compile":          bool(cfg.get("torch_compile", False)),
            "non_blocking_transfer":  bool(cfg.get("non_blocking_transfer", False)),
        }

        # ── cuDNN benchmark / deterministic ──────────────────────────
        if cls._train_opt_config["cudnn_benchmark"] and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    @classmethod
    def is_optimization_enabled(cls, key: str) -> bool:
        """Return ``True`` if the named train optimization is active."""
        return bool(cls._train_opt_config.get(key, False))

    @classmethod
    def get_optimization_status(cls) -> Dict[str, Any]:
        """
        Return a dict describing the current optimization state,
        suitable for logging during ``prepare()``.
        """
        result: Dict[str, Any] = {}
        for key, enabled in cls._train_opt_config.items():
            entry: Dict[str, Any] = {"enabled": enabled}
            if enabled and key in _BIT_EXACT_SENSITIVE_OPTS:
                entry["warning"] = "May affect bit-exact reproducibility"
            # torch_compile extra check
            if key == "torch_compile" and enabled:
                if not hasattr(torch, "compile"):
                    entry["enabled"] = False
                    entry["warning"] = "torch.compile not available (PyTorch < 2.0) — disabled"
            result[key] = entry
        return result

    @staticmethod
    def to_device(tensor: torch.Tensor,
                  device: torch.device,
                  non_blocking: Optional[bool] = None) -> torch.Tensor:
        """
        Move *tensor* to *device*, respecting the global
        ``non_blocking_transfer`` optimisation flag.

        When *non_blocking* is ``None`` (default), the global flag is used.
        """
        if non_blocking is None:
            non_blocking = TrainingUtils.is_optimization_enabled("non_blocking_transfer")
        if non_blocking and device.type == "cuda":
            return tensor.to(device, non_blocking=True)
        return tensor.to(device)

    @staticmethod
    def resolve_amp_dtype(device: torch.device) -> Optional[torch.dtype]:
    # ------------------------------------------------------------------
    # AMP helpers
    # ------------------------------------------------------------------
        """
        Choose the best AMP dtype for the given device.
        - CUDA + BF16 supported (Ampere+: A10G, L4, A100, H100) → bfloat16
        - CUDA, no BF16 (V100, T4 …)                             → FP32 (no AMP)
        - MPS                                                     → FP32 (no AMP)
        - CPU                                                     → FP32 (no AMP)
        Returns None when AMP should not be used.
        """
        dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
        if dev_type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None  # fallback: run in FP32, no AMP

    @staticmethod
    def make_autocast(device: torch.device, enabled: bool,
                      dtype: torch.dtype | None = None):
        """
        Return an autocast context for the given device.
        - If *dtype* is not given, auto-selects via resolve_amp_dtype.
        - BF16 on capable CUDA: uses torch.autocast (no GradScaler needed).
        - Falls back to contextlib.nullcontext (FP32) when BF16 not supported.
        """
        if not enabled:
            return contextlib.nullcontext()
        if dtype is None:
            dtype = TrainingUtils.resolve_amp_dtype(device)
        if dtype is None:  # device does not support AMP with safe dtype
            return contextlib.nullcontext()
        dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
        if dev_type in ("cuda", "cpu"):
            return torch.autocast(device_type=dev_type, dtype=dtype)
        return contextlib.nullcontext()

    @staticmethod
    def make_scaler(device: torch.device, enabled: bool,
                    dtype: torch.dtype | None = None):
        """
        Return a GradScaler only for FP16 CUDA (not needed for BF16).
        BF16 has the same exponent range as FP32, so no scaling is required.
        """
        if not enabled:
            return None
        if dtype is None:
            dtype = TrainingUtils.resolve_amp_dtype(device)
        # BF16 and FP32 do not need a scaler; only legacy FP16 does.
        if dtype == torch.float16:
            dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
            if dev_type == "cuda":
                return torch.cuda.amp.GradScaler(enabled=True)
        return None

    @staticmethod
    def set_seed_all(seed_input: int = 42):
        random.seed(seed_input)
        np.random.seed(seed_input)
        torch.manual_seed(seed_input)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_input)
            torch.cuda.manual_seed_all(seed_input)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def set_seed(seed_input: int = 42):
        random.seed(seed_input)
        np.random.seed(seed_input)
        torch.manual_seed(seed_input)

    @staticmethod
    def build_training_header(server_node: Any) -> Dict[str, Any]:
        """Collect common training header info from a server node safely."""
        cfg = getattr(getattr(server_node, "node_var", None), "config_dict", {}) or {}
        client_nodes = getattr(server_node, "client_nodes", []) or []

        def _get(path, default=None):
            cur = cfg
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            return cur

        rank_distribution_cfg = _get(["rank_distribution"], {}) or {}
        rank_ratio_list = _get(["rank_distribution", "rank_ratio_list"], [])
        client_yaml_snapshots = []
        effective_client_rank_ratios = []
        effective_client_share_model = []
        for index, client_node in enumerate(client_nodes):
            node_var = getattr(client_node, "node_var", None)
            client_cfg = getattr(node_var, "config_dict", {}) or {}
            nn_model_cfg = client_cfg.get("nn_model", {}) if isinstance(client_cfg, dict) else {}
            client_yaml_snapshots.append({
                "index": index,
                "node_id": getattr(client_node, "node_id", None),
                "yaml": client_cfg,
            })
            effective_client_rank_ratios.append(nn_model_cfg.get("rank_ratio"))
            effective_client_share_model.append(nn_model_cfg.get("share_model"))

        # --- Node topology ---
        client_nodes_cfg: dict = _get(["client_nodes"], {}) or {}
        total_clients = sum(
            int(g.get("number", 1)) if isinstance(g, dict) else 0
            for g in client_nodes_cfg.values()
        ) if client_nodes_cfg else None

        # --- Client selection ---
        client_selection_cfg: dict = _get(["client_selection"], {}) or {}

        # --- Hyperparameters (dedicated section) ---
        hyperparameters = {
            # FL schedule
            "training_rounds":        _get(["general", "training_rounds"]),
            "local_epochs":           _get(["training", "epochs"]),
            "total_clients":          total_clients,
            "selected_clients":       client_selection_cfg.get("number"),
            "client_selection_round": client_selection_cfg.get("round"),
            "client_selection_method": client_selection_cfg.get("method"),
            # Optimiser
            "optimizer":              _get(["optimizer", "type"]),
            "lr":                     _get(["optimizer", "lr"]),
            "momentum":               _get(["optimizer", "momentum"]),
            "weight_decay":           _get(["optimizer", "weight_decay"]),
            "nesterov":               _get(["optimizer", "nesterov"]),
            # Data
            "batch_size":             _get(["data_loader", "batch_size"]),
            "shuffle":                _get(["data_loader", "shuffle"]),
            # Trainer
            "trainer_type":           _get(["trainer", "trainer_type"]),
            # LoRA / rank (present only for LoRA experiments)
            "rank_ratio_list":        rank_ratio_list if rank_ratio_list else None,
        }
        # Remove keys whose value is None to keep the dict clean
        hyperparameters = {k: v for k, v in hyperparameters.items() if v is not None}

        return {
            # --- Experiment identity ---
            "general":          _get(["general"], {}),
            "dataset":          _get(["data_loader", "name"]),
            "model":            _get(["nn_model", "name"]),
            "loss_function":    _get(["loss_func", "type"]),
            "aggregation":      _get(["aggregation", "method"]),
            "client_selection": client_selection_cfg,
            # --- Training hyperparameters (dedicated section) ---
            "hyperparameters":  hyperparameters,
            # --- Effective YAML snapshot used by this run ---
            "effective_yaml": {
                "server_yaml": cfg,
                "client_yamls": client_yaml_snapshots,
            },
            "effective_client_rank_ratios": effective_client_rank_ratios,
            "effective_client_share_model": effective_client_share_model,
            # --- Rank / LoRA config (kept for backward compat) ---
            "rank_distribution":  rank_distribution_cfg,
            "rank_ratio_list":    rank_ratio_list,
            "rank_ratio_list_str": str(rank_ratio_list) if rank_ratio_list else "",
            # --- Legacy flat keys (kept for backward compat) ---
            "epoch":      _get(["training", "epochs"]),
            "batch_size": _get(["data_loader", "batch_size"]),
        }
