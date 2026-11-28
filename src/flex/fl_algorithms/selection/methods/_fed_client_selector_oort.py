from __future__ import annotations
import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Iterable, Tuple

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector

class FedClientSelector_Oort(FedClientSelector):
    """
    Oort client selection algorithm (OSDI).

    Global state (Algorithm 1):
        E: explored clients      -> clients with hist[cid]['count'] > 0
        U: per-client utility    -> hist[cid]['U']
        L: participation count   -> hist[cid]['count']  (used in UCB term)
        D: last duration         -> hist[cid]['D']
        R: round counter         -> self.round_idx
        T: preferred round duration (global pacer) -> self.T

    SelectParticipant(C, K, ε, T, α) is implemented by select().
    UpdateWithFeedback(E, U, L, D) is not explicitly called by external strategy but 
    in this framework we can hook into `select` or use internal data dictionary `_clients_data_dict`.
    Actually, this class needs to maintain history across `select` calls.
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "oort"

        # Oort hyperparams (defaults)
        self.epsilon = 0.1        # exploitation factor ε (fraction for exploration = ε)
        self.pacer_step = 5         # step window W (used for ΣU(R-2W:R-W) and ΣU(R-W:R))
        self.pacer_delta = 0.1    # T step Δ 
        self.alpha_penalty = 1.0  # straggler penalty exponent α
        self.cut_factor = 0.95    # c in CutOffUtil(E, c × Util((1−ε)K))
        self.ucb_const = 0.1      # 0.1 in UCB term

        # EWMA / Other
        self.util_alpha = 0.7     # U(i) EWMA (new value weight is (1 - util_alpha) in this implementation?)
                                  # Wait, in user code: h["U"] = self.util_alpha * h["U"] + (1.0 - self.util_alpha) * u_obs
        self.min_time = 1e-6      # duration prevent div 0
        self.init_T = 1.0         # T initial value

        # Override defaults if present in args.config_dict (assuming args has a config_dict)
        # Note: FedClientSelectorArgs might not expose raw config dict easily depending on implementation.
        # Assuming we stick to defaults or modify args later.
        
        # Global Vars: R, T, U_history
        self.round_idx: int = 0      # R
        self.T: float = float(self.init_T)  # T
        self._round_util_history: List[float] = []  # save round ΣU for pacer

        # Client History: cid -> {'U', 'count', 'D', 'last_round'}
        self.hist: Dict[str, Dict[str, float]] = {}

        self._rng = random.Random(self._args.random_seed if self._args else 42)
        np.random.seed(self._args.random_seed if self._args else 42)

    def select(self, client_list: list, select_number: int = -1) -> list:
        """
        Correspond to Algorithm 1 SelectParticipant(C, K, ε, T, α).
        """
        # 1. Update history based on available data from previous rounds.
        # `_clients_data_dict` is expected to contain latest feedback.
        if self._clients_data_dict:
            client_info = []
            for cid, data in self._clients_data_dict.items():
                if isinstance(data, dict):
                    row = dict(data)
                else:
                    row = {"value": data}
                row.setdefault("client_id", str(cid))
                client_info.append(row)
            self.update_history(client_info=client_info, participated_ids=list(self._clients_data_dict.keys()))
            
            # Clear data dict after processing to avoid re-processing same feedback in next call
            self._clients_data_dict = {}

        # 2. Perform selection
        if select_number <= 0:
            select_number = self._args.select_number

        name_dict = {str(c.node_id): str(getattr(c, "name", c.node_id)) for c in client_list}
        selected_ids = self.select_clients(
            metric_matrix=self._clients_data_dict,
            name_dict=name_dict,
            number=select_number
        )

        return [c for c in client_list if str(c.node_id) in set(selected_ids)]

    # ------------------------------------------------------------------ #
    # Internal: Update History Logic
    # ------------------------------------------------------------------ #
    def _update_history_from_dict(self, clients_data_dict: dict):
        """
        Update history using the dictionary usually passed via `with_clients_data`.
        Iterate over clients who have data (meaning they participated recently).
        """
        # We need to distinguish which round the data belongs to.
        # The framework's `_clients_data_dict` might persist data.
        # Oort needs fresh feedback.
        # Assuming `_clients_data_dict` contains ONLY the latest updates from participated clients 
        # or we check `last_round` to avoid double counting if possible. 
        # But `_clients_data_dict` structure isn't fully defined here. 
        # We will assume it contains latest 'train_record'.
        
        round_sum_U = 0.0
        updated_count = 0

        for cid, data in clients_data_dict.items():
            if not isinstance(data, dict): continue
            
            # Simple check if this is "new" data? 
            # In this framework context, usually `select` is called at start of round R.
            # `with_clients_data` is called with results from Round R-1.
            # So we process it now.
            
            row = pd.Series(data) # Wrap in Series for easy access helpers? Or just dict access.
            # Flattener helper might be needed if data is nested
            flat_row = self._flatten_data(data)
            
            # Ensure hist entry
            if cid not in self.hist:
                self.hist[cid] = {
                    "U": 0.0,
                    "count": 0.0,
                    "D": 1.0,
                    "last_round": -1.0,
                }
            h = self.hist[cid]

            # We need to update ONLY if we haven't processed this specific update yet.
            # Since we don't have a reliable "update ID", we might just process everything
            # and rely on the fact that `_clients_data_dict` is refreshed by the strategy.
            
            # --- Observe Utility U_obs(i) ---
            u_obs = self._observe_util(flat_row)
            
            # If u_obs is 0, maybe they didn't participate? 
            # But loss reduction could be 0.
            # Let's proceed.

            # --- EWMA Update U(i) ---
            # h["U"] = self.util_alpha * h["U"] + (1.0 - self.util_alpha) * u_obs
            # NOTE: User requested code structure:
            h["U"] = self.util_alpha * h["U"] + (1.0 - self.util_alpha) * u_obs

            # --- Update L(i) and duration D(i) ---
            # We assume presence in `clients_data_dict` implies participation.
            h["count"] = h.get("count", 0.0) + 1.0
            h["D"] = float(flat_row.get("latency", flat_row.get("duration", 1.0)))
            h["last_round"] = float(self.round_idx)  # Record current/prev round?

            self.hist[cid] = h
            round_sum_U += h["U"]
            updated_count += 1

        # Record round sum utility for Pacer
        if updated_count > 0 and round_sum_U > 0.0:
            self._round_util_history.append(round_sum_U)


    def _flatten_data(self, data: dict) -> dict:
        """Helper to flatten nested dictionary (e.g. train_record) to single level."""
        flat = dict(data)
        if "train_record" in data and isinstance(data["train_record"], dict):
            flat.update(data["train_record"])
        return flat

    # ------------------------------------------------------------------ #
    # Internal: Selection Logic (from user's SelectParticipant replacement)
    # ------------------------------------------------------------------ #
    def select_clients(
        self,
        metric_matrix: Any,
        name_dict: Dict[str, str],
        number: int,
        break_client_set: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Line-by-line reconstruction of Algorithm 1 (SelectParticipant),
        ignoring device performance metrics.
        """
        if number <= 0:
            self._print_selected_clients([], name_dict)
            return []

        # line 5: R ← R + 1
        self.round_idx += 1
        R = max(1, self.round_idx)

        eps = float(kwargs.get("epsilon", kwargs.get("eps", self.epsilon)))
        eps = min(max(eps, 0.0), 1.0)
        K = int(max(0, number))

        break_set = {str(x) for x in (break_client_set or [])}

        C_ids = [str(cid) for cid in name_dict.keys() if str(cid) not in break_set]
        if not C_ids:
            self._print_selected_clients([], name_dict)
            return []

        # Init hist for new clients
        for cid in C_ids:
            if cid not in self.hist:
                self.hist[cid] = {
                    "U": 0.0,
                    "count": 0.0,
                    "D": 1.0,
                    "last_round": -1.0,
                }

        # -------------------- line 7–8: Pacer ------------------------------
        self._pacer()

        # -------------------- Exploitation #1: line 9–12 -------------------
        E = [cid for cid in C_ids if self.hist[cid]["count"] > 0.0]

        Util: Dict[str, float] = {}
        for cid in E:
            h = self.hist[cid]
            U_i = float(h["U"])
            L_i = max(1.0, float(h["count"]))
            ucb = math.sqrt(self.ucb_const * math.log(R) / L_i)
            util_i = U_i + ucb
            Util[cid] = max(0.0, float(util_i))

        # -------------------- Exploitation #2: line 13–15 ------------------
        k_exploit = min(len(E), int(round(K * (1.0 - eps))))
        picked_exploit: List[str] = []

        if k_exploit > 0 and Util:
            sorted_ids = sorted(Util.keys(), key=lambda cid: Util[cid], reverse=True)
            idx_cut = min(max(k_exploit - 1, 0), len(sorted_ids) - 1)
            cutoff_val = Util[sorted_ids[idx_cut]] * self.cut_factor
            W_set = [cid for cid in sorted_ids if Util[cid] >= cutoff_val]

            k_sample = min(k_exploit, len(W_set))
            if k_sample > 0:
                weights = np.array([Util[cid] for cid in W_set], dtype=float)
                weights = weights / max(weights.sum(), 1e-8)
                picked_exploit = list(
                    np.random.choice(W_set, size=k_sample, replace=False, p=weights)
                )

        # -------------------- Exploration: line 16 -------------------------
        remaining_K = max(0, K - len(picked_exploit))
        picked_explore: List[str] = []
        if remaining_K > 0:
            unexplored = [cid for cid in C_ids if cid not in E]
            if unexplored:
                k_explore = min(remaining_K, int(round(eps * K)) or remaining_K, len(unexplored))
                picked_explore = list(
                    np.random.choice(unexplored, size=k_explore, replace=False)
                )

        P = picked_exploit + picked_explore

        if len(P) < K:
            remaining = [cid for cid in C_ids if cid not in P]
            self._rng.shuffle(remaining)
            P.extend(remaining[: K - len(P)])

        self._print_selected_clients(P, name_dict)
        return P

    def update_history(
        self,
        client_info: List[Dict[str, Any]],
        participated_ids: Optional[List[str]] = None,
        round_idx: Optional[int] = None,
    ) -> None:
        """
        Update with feedback (Algorithm 1), ignoring device performance metrics.
        """
        if round_idx is not None:
            self.round_idx = int(round_idx)

        part_set = set(map(str, participated_ids or []))
        round_sum_U = 0.0

        for item in client_info:
            cid = str(item.get("client_id", item.get("id", "")))
            if not cid:
                continue
            if part_set and cid not in part_set:
                continue

            row = pd.Series(item)
            if cid not in self.hist:
                self.hist[cid] = {
                    "U": 0.0,
                    "count": 0.0,
                    "D": 1.0,
                    "last_round": -1.0,
                }

            h = self.hist[cid]
            u_obs = self._observe_util(row)
            h["U"] = self.util_alpha * h["U"] + (1.0 - self.util_alpha) * u_obs
            h["count"] = h.get("count", 0.0) + 1.0
            h["D"] = 1.0
            h["last_round"] = float(self.round_idx)

            self.hist[cid] = h
            round_sum_U += h["U"]

        if round_sum_U > 0.0:
            self._round_util_history.append(round_sum_U)

    def _print_selected_clients(self, selected: List[str], name_dict: Dict[str, str]) -> None:
        """Debug helper (suppressed by default)."""
        pass

    # ------------------------------------------------------------------ #
    # Pacer
    # ------------------------------------------------------------------ #
    def _pacer(self) -> None:
        W = self.pacer_step
        if W <= 0: return
        history = self._round_util_history
        if len(history) < 2 * W: return

        prev_sum = sum(history[-2 * W : -W])
        curr_sum = sum(history[-W:])

        if prev_sum > curr_sum:
            self.T += self.pacer_delta

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _observe_util(self, row: dict) -> float:
        """
        Construct U_obs(i): "Higher is better".
        Heuristics:
          - (loss_before - loss_after)
          - Or directly oort_statistical_utility if available
        """
        # If Oort provided utility is already computed
        if "oort_statistical_utility" in row:
             return float(row["oort_statistical_utility"])

        # Else heuristic
        loss_before = float(row.get("loss_before", row.get("prev_loss", 0.0)) or 0.0)
        # Fallback: if train_loss_before exists
        if loss_before == 0.0 and "train_loss_before" in row:
             loss_before = float(row["train_loss_before"])
             
        # "train_loss" or "loss" usually means current loss after training
        loss_after = float(row.get("train_loss", row.get("loss", row.get("loss_after", loss_before))) or loss_before)
        
        delta_loss = max(0.0, loss_before - loss_after)
        
        if delta_loss == 0.0:
            # If no reduction info, use current loss as statistical utility
            return float(loss_after)
            
        return float(delta_loss)

    def _extract_time_cost(self, row: dict, default: float = 1.0) -> float:
        for key in ("time_cost", "total_time_cost", "round_time", "training_time", "train_time", "duration"):
            if key in row and row[key] is not None:
                try:
                    v = float(row[key])
                    if np.isfinite(v):
                        return max(self.min_time, v)
                except (TypeError, ValueError):
                    continue
        return max(self.min_time, float(default))

