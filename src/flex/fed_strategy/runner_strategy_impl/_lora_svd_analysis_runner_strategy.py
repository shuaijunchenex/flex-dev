from __future__ import annotations

import json
import os

import numpy as np
import torch
from ...ml_utils.tqdm_utils import pbar

from ...fed_node import FedNodeClient, FedNodeServer
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy
from ...fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...ml_utils.training_utils import TrainingUtils


class LoraSvdAnalysisRunnerStrategy(RunnerStrategy):
    """
    Runner strategy for analyzing singular-value (eigenvalue) decay of aggregated LoRA matrices.

    After every aggregation round the strategy performs the following analysis and saves the
    results of **each round** to a separate JSON file inside `output_dir`:

    1. Records ``W_b + lora_A @ lora_B``  (the effective / merged weight, called WbAB)
       for every LoRA layer found in the aggregated global state-dict.
    2. Records the global average gradient, computed as the data-volume-weighted mean of
       ``(global_weight - client_updated_weight)`` across **all** participating clients and
       **all** their local epochs (the multi-epoch update is already absorbed in the weight
       difference returned by each client's trainer).
    3. Performs SVD on:
          • WbAB per layer        → sorted singular values (descending)
          • lora_A @ lora_B per layer  (pure LoRA contribution)
          • avg_gradient_lora_A @ avg_gradient_lora_B
    4. Saves each round to  ``<output_dir>/round_XXXX.json``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        runner: FedRunner,
        args: StrategyArgs,
        client_nodes,
        server_node,
        output_dir: str = "./lora_svd_analysis_output",
    ) -> None:
        super().__init__(runner)

        self._strategy_type = "lora_svd"
        self.args = args
        self.client_nodes: list[FedNodeClient] = client_nodes
        self.server_node: FedNodeServer = server_node

        # Allow output_dir to be set from the yaml config
        output_dir_cfg = args.key_value_dict.data.get("lora_svd_output_dir", None)
        self.output_dir: str = output_dir_cfg if output_dir_cfg else output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self.set_node_connection()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def set_node_connection(self) -> None:
        self.server_node.set_client_nodes(self.client_nodes)
        for client in self.client_nodes:
            client.set_server_node(self.server_node)

    # ------------------------------------------------------------------
    # Abstract method implementations required by RunnerStrategy
    # ------------------------------------------------------------------
    def _create_inner(self, client_nodes=None, server_nodes=None) -> None:
        return self

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast()

    def simulate_server_update_process(self, weight=None):
        if weight is not None:
            self.server_node.strategy.server_update(weight)

    # ------------------------------------------------------------------
    # Simulation helpers (mirrors SpRunnerStrategy)
    # ------------------------------------------------------------------
    def simulate_client_local_training_process(self, participants):
        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.strategy.run_local_training()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record,
            }

    # ------------------------------------------------------------------
    # LoRA key discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _find_lora_layers(state_dict: dict) -> dict[str, dict[str, str | None]]:
        """
        Scan *state_dict* and return a mapping::

            { layer_prefix -> { 'lora_A': key, 'lora_B': key, 'weight': key | None } }

        Works for RBLA-style aggregated weights that still contain lora_A / lora_B keys.
        """
        lora_A_keys: dict[str, str] = {}
        lora_B_keys: dict[str, str] = {}

        for key in state_dict:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "lora_A":
                    prefix = ".".join(parts[:i])
                    lora_A_keys[prefix] = key
                    break
                if part == "lora_B":
                    prefix = ".".join(parts[:i])
                    lora_B_keys[prefix] = key
                    break

        layers: dict[str, dict[str, str | None]] = {}
        for prefix, a_key in lora_A_keys.items():
            if prefix not in lora_B_keys:
                continue
            b_key = lora_B_keys[prefix]
            w_key = f"{prefix}.weight" if f"{prefix}.weight" in state_dict else None
            layers[prefix] = {"lora_A": a_key, "lora_B": b_key, "weight": w_key}

        return layers

    @staticmethod
    def _find_sp_layers(state_dict: dict, sp_suffix: str = "sp_aggregated") -> dict[str, dict[str, str | None]]:
        """
        Scan SP-style aggregated *state_dict* and return a mapping::

            { layer_prefix -> { 'sp_aggregated': key, 'weight': key | None } }

        SP aggregation produces keys like ``prefix.sp_aggregated`` (= W_b + ΔW)
        instead of lora_A / lora_B.
        """
        layers: dict[str, dict[str, str | None]] = {}
        for key in state_dict:
            if key.endswith(f".{sp_suffix}"):
                prefix = key[: -(len(sp_suffix) + 1)]
                w_key = f"{prefix}.weight" if f"{prefix}.weight" in state_dict else None
                layers[prefix] = {"sp_aggregated": key, "weight": w_key}
        return layers

    # ------------------------------------------------------------------
    # Matrix computation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_WbAB(state_dict: dict, layers: dict) -> dict[str, torch.Tensor]:
        """
        Return ``{ prefix: W_b + lora_A @ lora_B }`` (float32, CPU).
        Handles both RBLA format (lora_A/lora_B keys) and SP format (sp_aggregated key).
        """
        result: dict[str, torch.Tensor] = {}
        for prefix, keys in layers.items():
            if "sp_aggregated" in keys:
                # SP: sp_aggregated already IS W_b + ΔW
                result[prefix] = state_dict[keys["sp_aggregated"]].float().cpu()
            else:
                A = state_dict[keys["lora_A"]].float().cpu()
                B = state_dict[keys["lora_B"]].float().cpu()
                AB = A @ B
                if keys["weight"] is not None:
                    Wb = state_dict[keys["weight"]].float().cpu()
                    result[prefix] = Wb + AB
                else:
                    result[prefix] = AB
        return result

    @staticmethod
    def _compute_AB(state_dict: dict, layers: dict) -> dict[str, torch.Tensor]:
        """
        Return ``{ prefix: lora_A @ lora_B }`` (LoRA contribution only).
        For SP format: returns sp_aggregated - weight (ΔW only); if weight missing, returns sp_aggregated.
        """
        result: dict[str, torch.Tensor] = {}
        for prefix, keys in layers.items():
            if "sp_aggregated" in keys:
                sp_mat = state_dict[keys["sp_aggregated"]].float().cpu()
                if keys["weight"] is not None:
                    Wb = state_dict[keys["weight"]].float().cpu()
                    result[prefix] = sp_mat - Wb
                else:
                    result[prefix] = sp_mat
            else:
                A = state_dict[keys["lora_A"]].float().cpu()
                B = state_dict[keys["lora_B"]].float().cpu()
                result[prefix] = B @ A
        return result

    # ------------------------------------------------------------------
    # Real-gradient collection (from ModelTrainer_LoraGrad)
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_avg_real_gradients(
        client_updates: list,
    ) -> dict[str, dict[str, np.ndarray | None]]:
        """
        Collect data-volume-weighted mean of the real gradients stored in
        ``train_record["lora_grad"]`` by ``ModelTrainer_LoraGrad``.

        Returns an empty dict when no client carries ``lora_grad`` data.
        """
        total_weight = 0.0
        grad_accum: dict[str, dict[str, np.ndarray | None]] = {}

        for update in client_updates:
            tr = update["train_record"]
            lora_grad: dict = tr.get("lora_grad", {})
            if not lora_grad:
                continue

            vol: float = float(tr.get("data_sample_num", 1))

            for prefix, gd in lora_grad.items():
                if prefix not in grad_accum:
                    grad_accum[prefix] = {"lora_A": None, "lora_B": None}
                for key in ("lora_A", "lora_B"):
                    g = gd.get(key)
                    if g is None:
                        continue
                    g_arr = g if isinstance(g, np.ndarray) else np.array(g, dtype=np.float32)
                    weighted = g_arr * vol
                    prev = grad_accum[prefix][key]
                    grad_accum[prefix][key] = weighted if prev is None else prev + weighted

            total_weight += vol

        if total_weight == 0 or not grad_accum:
            return {}

        result: dict[str, dict[str, np.ndarray | None]] = {}
        for prefix, gd in grad_accum.items():
            result[prefix] = {
                key: (gd[key] / total_weight) if gd[key] is not None else None
                for key in ("lora_A", "lora_B")
            }
        return result

    # ------------------------------------------------------------------
    # Average gradient computation (weight-diff proxy fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_average_gradient(
        global_weight: dict,
        client_updates: list,
        layers: dict,
    ) -> dict[str, dict[str, np.ndarray | None]]:
        """
        Compute data-volume-weighted average gradient for LoRA parameters.

        ``gradient_proxy = global_weight[param] - client_updated_weight[param]``

        (For SGD with lr and n steps: w_after = w_before - lr * Σ grad,
        so w_before - w_after ∝ accumulated gradient.)

        Returns::

            { prefix -> { 'lora_A': np.ndarray | None,
                          'lora_B': np.ndarray | None } }
        """
        if not client_updates:
            return {}

        def _aligned_diff(
            global_tensor: torch.Tensor,
            updated_tensor: torch.Tensor,
            lora_key_name: str,
        ) -> torch.Tensor | None:
            """Return ``global - updated`` with lightweight shape adaptation.

            Strategy:
              1) direct subtraction when shapes match,
              2) try updated.T,
              3) for heterogeneous LoRA ranks, pad/truncate updated to global shape,
              4) otherwise return None (caller decides to skip).
            """
            g = global_tensor.float().cpu()
            u = updated_tensor.float().cpu()

            if g.shape == u.shape:
                return g - u

            if g.ndim == 2 and u.ndim == 2 and g.shape == u.transpose(0, 1).shape:
                return g - u.transpose(0, 1)

            # Heterogeneous rank support:
            #   lora_A: [r, in]
            #   lora_B: [out, r]
            if g.ndim == 2 and u.ndim == 2:
                if lora_key_name == "lora_A" and g.shape[1] == u.shape[1]:
                    # Align rank dimension (dim=0)
                    if u.shape[0] < g.shape[0]:
                        pad_rows = g.shape[0] - u.shape[0]
                        u = torch.cat(
                            [u, torch.zeros((pad_rows, u.shape[1]), dtype=u.dtype)],
                            dim=0,
                        )
                    elif u.shape[0] > g.shape[0]:
                        u = u[: g.shape[0], :]
                    return g - u

                if lora_key_name == "lora_B" and g.shape[0] == u.shape[0]:
                    # Align rank dimension (dim=1)
                    if u.shape[1] < g.shape[1]:
                        pad_cols = g.shape[1] - u.shape[1]
                        u = torch.cat(
                            [u, torch.zeros((u.shape[0], pad_cols), dtype=u.dtype)],
                            dim=1,
                        )
                    elif u.shape[1] > g.shape[1]:
                        u = u[:, : g.shape[1]]
                    return g - u

            return None

        total_weight = 0.0
        grad_accum: dict[str, dict[str, torch.Tensor | None]] = {}
        warned_mismatch: set[tuple[str, str]] = set()

        for update in client_updates:
            tr = update["train_record"]
            updated_w: dict = tr["updated_weights"]
            vol: float = float(tr.get("data_sample_num", 1))

            for prefix, keys in layers.items():
                if prefix not in grad_accum:
                    grad_accum[prefix] = {"lora_A": None, "lora_B": None}

                for lora_key_name in ("lora_A", "lora_B"):
                    sd_key = keys[lora_key_name]
                    if sd_key in updated_w and sd_key in global_weight:
                        aligned = _aligned_diff(global_weight[sd_key], updated_w[sd_key], lora_key_name)
                        if aligned is None:
                            warn_token = (prefix, lora_key_name)
                            if warn_token not in warned_mismatch:
                                console.warn(
                                    "[LoRA SVD Analysis] Skip gradient diff for "
                                    f"'{prefix}.{lora_key_name}': shape mismatch "
                                    f"global={tuple(global_weight[sd_key].shape)} vs "
                                    f"updated={tuple(updated_w[sd_key].shape)}"
                                )
                                warned_mismatch.add(warn_token)
                            continue

                        diff = aligned * vol
                        prev = grad_accum[prefix][lora_key_name]
                        grad_accum[prefix][lora_key_name] = diff if prev is None else prev + diff

            total_weight += vol

        if total_weight == 0:
            return {}

        avg_gradient: dict[str, dict[str, np.ndarray | None]] = {}
        for prefix, grads in grad_accum.items():
            avg_gradient[prefix] = {
                lora_key_name: (
                    (grads[lora_key_name] / total_weight).numpy()
                    if grads[lora_key_name] is not None
                    else None
                )
                for lora_key_name in ("lora_A", "lora_B")
            }

        return avg_gradient

    # ------------------------------------------------------------------
    # SVD helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_singular_values(matrices: dict[str, torch.Tensor | np.ndarray]) -> dict[str, list[float]]:
        """
        Compute SVD for each matrix and return **sorted (descending)** singular values.
        """
        results: dict[str, list[float]] = {}
        for prefix, mat in matrices.items():
            mat_np = mat.numpy() if isinstance(mat, torch.Tensor) else mat
            try:
                sv = np.linalg.svd(mat_np, compute_uv=False)   # sorted descending by numpy
                results[prefix] = sv.tolist()
            except Exception as exc:
                console.error(f"[LoRA SVD Analysis] SVD failed for '{prefix}': {exc}")
                results[prefix] = []
        return results

    @staticmethod
    def _compose_lora_delta(
        gA: np.ndarray,
        gB: np.ndarray,
        target_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Compose LoRA delta from gradient factors with robust orientation handling.

        Tries both ``A @ B`` and ``B @ A`` (and picks the one that matches
        ``target_shape`` when available). Raises ValueError when no valid composition exists.
        """
        candidates: list[np.ndarray] = []

        if gA.ndim == 2 and gB.ndim == 2:
            if gA.shape[1] == gB.shape[0]:
                candidates.append(gA @ gB)
            if gB.shape[1] == gA.shape[0]:
                candidates.append(gB @ gA)

        if not candidates:
            raise ValueError(
                f"No valid LoRA gradient composition path for shapes A={gA.shape}, B={gB.shape}"
            )

        if target_shape is not None:
            for mat in candidates:
                if mat.shape == target_shape:
                    return mat

        return candidates[0]

    # ------------------------------------------------------------------
    # Probe-training SVD helper
    # ------------------------------------------------------------------
    def _probe_global_svd(
        self,
        round_idx: int,
        probe_epochs: int = 1,
    ) -> dict[str, list[float]]:
        """
        Ask the server strategy to train only the base weight W (lora_A / lora_B
        frozen) on a copy of the global model for *probe_epochs* epochs and
        return SVD singular values of the mean gradient of each W matrix.

        Returns ``{ layer_prefix: [sv, ...] }`` or empty dict on failure.
        """
        try:
            grad_W = self.server_node.strategy.probe_train_global_model(
                probe_epochs=probe_epochs
            )
        except AttributeError:
            console.warn(
                "[LoRA SVD Analysis] Server strategy does not support "
                "probe_train_global_model(); skipping probe SVD."
            )
            return {}

        if not grad_W:
            return {}

        results: dict[str, list[float]] = {}
        for param_name, grad_arr in grad_W.items():
            # param_name e.g. "_fc1.weight" → prefix "_fc1"
            prefix = param_name[: -len(".weight")] if param_name.endswith(".weight") else param_name
            try:
                sv = np.linalg.svd(grad_arr, compute_uv=False)
                results[prefix] = sv.tolist()
            except Exception as exc:
                console.error(
                    f"[LoRA SVD Analysis] Probe W-grad SVD failed for '{prefix}': {exc}"
                )
                results[prefix] = []
        return results

    # ------------------------------------------------------------------
    # Per-round analysis
    # ------------------------------------------------------------------
    def _analyze_round(
        self,
        round_idx: int,
        global_weight_before: dict,
        aggregated_weight: dict,
        client_updates: list,
    ) -> None:
        """Run the full analysis pipeline for one training round and save results.

        Supports two aggregated-weight formats:
          • RBLA: aggregated_weight contains ``lora_A`` / ``lora_B`` keys.
          • SP:   aggregated_weight contains ``sp_aggregated`` keys (= W_b + ΔW).

        In both cases:
          1. WbAB matrix per layer is extracted.
          2. LoRA-only contribution (ΔW = A@B) per layer is extracted.
          3. Average gradient is computed from the *client* lora_A/lora_B diffs
             (client updates always carry LoRA A/B regardless of aggregation style).
        """
        # Auto-detect format
        sp_layers = self._find_sp_layers(aggregated_weight)
        lora_layers = self._find_lora_layers(aggregated_weight)

        if sp_layers:
            layers = sp_layers
            console.info(f"[LoRA SVD Analysis] Round {round_idx}: SP format detected, {len(layers)} layer(s).")
        elif lora_layers:
            layers = lora_layers
            console.info(f"[LoRA SVD Analysis] Round {round_idx}: RBLA format detected, {len(layers)} layer(s).")
        else:
            console.warn(f"[LoRA SVD Analysis] Round {round_idx}: no LoRA/SP layers found, skipping.")
            return

        # 1. Compute effective matrices (W_b + ΔW and ΔW only)
        WbAB_mats = self._compute_WbAB(aggregated_weight, layers)
        AB_mats   = self._compute_AB(aggregated_weight, layers)

        # 2. Average gradient — use real gradients from train_stats["lora_grad"] when
        #    available (requires ModelTrainer_LoraGrad), otherwise fall back to the
        #    weight-difference proxy.
        avg_grad = self._collect_avg_real_gradients(client_updates)
        if not avg_grad:
            client_lora_layers = {}
            if client_updates:
                sample_update = client_updates[0]["train_record"]["updated_weights"]
                client_lora_layers = self._find_lora_layers(sample_update)
            avg_grad = self._compute_average_gradient(
                global_weight_before, client_updates, client_lora_layers
            )
            if avg_grad:
                console.info(
                    f"[LoRA SVD Analysis] Round {round_idx}: using weight-diff proxy for gradients."
                )
        else:
            console.info(
                f"[LoRA SVD Analysis] Round {round_idx}: using real gradients from trainer."
            )

        # 3. SVD singular values
        WbAB_svd = self._svd_singular_values(WbAB_mats)
        AB_svd   = self._svd_singular_values(AB_mats)

        # 3b. Probe-training SVD: train W only (lora_A/B frozen) for 3 epochs, record grad_W SVD
        probe_W_grad_svd = self._probe_global_svd(round_idx, probe_epochs=1)

        grad_AB_svd: dict[str, list[float]] = {}
        for prefix, grads in avg_grad.items():
            gA, gB = grads.get("lora_A"), grads.get("lora_B")
            if gA is not None and gB is not None:
                try:
                    target_shape = AB_mats[prefix].shape if prefix in AB_mats else None
                    grad_delta = self._compose_lora_delta(gA, gB, target_shape)
                    sv = np.linalg.svd(grad_delta, compute_uv=False)
                    grad_AB_svd[prefix] = sv.tolist()
                except Exception as exc:
                    console.error(f"[LoRA SVD Analysis] Gradient SVD failed for '{prefix}': {exc}")
                    grad_AB_svd[prefix] = []

        # 4. Build the serialisable record
        round_record: dict = {
            "round": round_idx,
            "aggregation_format": "sp" if sp_layers else "rbla",
            "num_participating_clients": len(client_updates),
            "lora_layers": list(layers.keys()),
            # ---- SVD singular values (descending) ----
            "WbAB_singular_values":              WbAB_svd,
            "AB_singular_values":                AB_svd,
            "avg_gradient_AB_singular_values":   grad_AB_svd,
            # ---- Probe W-gradient SVD (W trained 3 epochs, lora_A/B frozen, no weight update) ----
            "probe_W_grad_singular_values":      probe_W_grad_svd,
            # ---- Raw average gradient matrices ----
            # "avg_gradient_lora_A": {
            #     p: v["lora_A"].tolist() if v.get("lora_A") is not None else None
            #     for p, v in avg_grad.items()
            # },
            # "avg_gradient_lora_B": {
            #     p: v["lora_B"].tolist() if v.get("lora_B") is not None else None
            #     for p, v in avg_grad.items()
            # },
        }

        self._save_round_results(round_idx, round_record)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    def _save_round_results(self, round_idx: int, record: dict) -> None:
        filename = os.path.join(self.output_dir, f"round_{round_idx:04d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        console.info(f"[LoRA SVD Analysis] Saved → {filename}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def prepare(self, logger_header: str) -> None:
        self.server_node.prepare(logger_header, self.client_nodes)

    def run(self) -> None:
        console.out("Running [LoRA SVD Analysis] strategy...")
        header_data = TrainingUtils.build_training_header(self.server_node)
        self.server_node.prepare(header_data, self.client_nodes)
        # NOTE: No initial broadcast here — SP clients use their own initial model_weight
        # (LoRA state dict). The first broadcast happens AFTER the first aggregation,
        # at which point model_weight contains proper sp_aggregated keys.

        total_rounds: int = int(self.args.key_value_dict.data["training_rounds"])

        for round_idx in pbar(range(total_rounds + 1)):
            console.out(
                f"\n{'='*10} Training round {round_idx}/{total_rounds}, "
                f"Total participants: {len(self.client_nodes)} {'='*10}"
            )

            self.participants = self.server_node.select_clients(self.client_nodes)
            console.info(f"Round: {round_idx}, Select {len(self.participants)} clients: ").ok(
                f"{', '.join(map(str, self.participants))}"
            )

            # Snapshot global weight BEFORE local training (needed for gradient computation).
            # Use a participant client's model_weight (always in lora_A/lora_B format) rather
            # than the server's model_weight, which may be in sp_aggregated format after the
            # first round's apply_weight() — causing gradient diffs to silently return None.
            if self.participants:
                _ref_client = self.participants[0]
                global_weight_before: dict = {
                    k: v.clone() for k, v in _ref_client.node_var.model_weight.items()
                }
            else:
                global_weight_before: dict = {
                    k: v.clone() for k, v in self.server_node.node_var.model_weight.items()
                }

            # Clients train locally
            client_updates = list(self.simulate_client_local_training_process(self.participants))

            # Server aggregates
            self.server_node.receive_client_updates(client_updates)
            self.server_node.aggregation()
            self.server_node.apply_weight()

            # ──────────────────────────────────────────────────────────
            # SVD Analysis (runs after aggregation, before broadcast)
            # ──────────────────────────────────────────────────────────
            self._analyze_round(
                round_idx=round_idx,
                global_weight_before=global_weight_before,
                aggregated_weight=self.server_node.node_var.aggregated_weight,
                client_updates=client_updates,
            )

            # Broadcast new global model and evaluate
            self.server_node.broadcast()
            self.server_node.evaluate()
            self.server_node.record_evaluation()

            console.out(f"{'='*10} Round {round_idx}/{total_rounds} End {'='*10}")
