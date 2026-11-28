from __future__ import annotations

import dataclasses
import math
import random
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector
from ....ml_utils import console


# Client State Dataclass for PyramidFL
@dataclasses.dataclass
class _ClientState:
    """Client state tracking for PyramidFL selection
    
    Attributes:
        util_stat_ema: EWMA of statistical utility (loss reduction/accuracy gain)
        util_sys_ema: EWMA of system utility (throughput = 1 / time cost)
        avail_ema: EWMA of client availability (1 if observed, 0 otherwise)
        last_time: Most recent round time cost of the client
        last_seen_round: Last global round the client was observed
        participation_count: Number of times the client was selected (for exploration)
        stat_mean: Mean of historical statistical feedback (for exploration score)
        stat_std: Standard deviation of historical statistical feedback (for exploration score)
        stat_history: Historical record of statistical feedback values
    """
    util_stat_ema: float = 0.0
    util_sys_ema: float = 1.0
    avail_ema: float = 0.0
    last_time: float = 1.0
    last_seen_round: int = 0
    participation_count: int = 0
    stat_mean: float = 0.0
    stat_std: float = 0.0
    stat_history: List[float] = dataclasses.field(default_factory=list)
    
    # Value cache for unconnected clients
    last_sample_num: float = 1.0
    last_shard_keep: float = 1.0
    last_selected_round: int = -1


class FedClientSelector_PyramidFL(FedClientSelector):
    """
    PyramidFL-inspired client selection — **data-only simulation mode**.

    This implementation is a data-only PyramidFL-inspired selector for simulation.
    It does **not** model system heterogeneity, stragglers, latency, throughput,
    preferred duration, or dropout.  It only uses data-related signals such as:
      - training loss / squared-loss summary
      - sample number
      - optional label distribution
      - historical statistical utility (EWMA)
      - participation count / anti-starvation exploration

    This is NOT a faithful full PyramidFL runtime.  System-related parameters
    (T, delta_T, alpha_straggler, sys_alpha, avail_alpha, a_dropout, b_dropout,
    beta, I_fix) are kept for backward compatibility but do **not** affect
    selection when ``pyramidfl_data_only=True`` (the default).

    ── Configuration (yaml) ──────────────────────────────────────────────────
      pyramidfl_data_only                  : bool   default True
      pyramidfl_explore_fraction           : float  exploration fraction (default ε)
      pyramidfl_stat_history_len           : int    max history length (default 10)
      pyramidfl_utility_epsilon            : float  small bonus for anti-collapse
      pyramidfl_use_label_diversity        : bool   default False
      pyramidfl_label_diversity_weight     : float  default 0.0
      pyramidfl_use_system_utility_in_explore : bool default False (data-only)
      pyramidfl_force_one_explore_when_k2  : bool   default True
    """

    def __init__(
        self,
        args: FedClientSelectorArgs|None = None,
        # Algorithm core hyperparameters (defaults) — kept for backward compat
        epsilon: float = 0.2,
        delta_T: float = 0.1,
        alpha_straggler: float = 1.0,
        a_dropout: float = 0.5,
        b_dropout: float = 1.0,
        # EWMA coefficients (keep-old weights)
        stat_alpha: float = 0.7,
        sys_alpha: float = 0.7,
        avail_alpha: float = 0.8,
        # Initial preferred round duration
        init_T: float = 1.0,
        min_time: float = 1e-3,
        # Client-side hyperparameters (for OptimizationAtClient)
        beta: float = 1.0,
        I_fix: int = 100,
    ) -> None:
        super().__init__(args)

        self._args = args if args is not None else FedClientSelectorArgs()

        # Exploit/Explore parameters - read from args if available, otherwise use defaults
        self.epsilon = float(self._args.get("epsilon", epsilon))
        self.delta_T = float(self._args.get("delta_T", delta_T))
        self.alpha_straggler = float(self._args.get("alpha_straggler", alpha_straggler))

        # Dropout bounds for client-side shard keep ratio [a, b] — backward compat
        self.a_dropout = float(self._args.get("a_dropout", a_dropout))
        self.b_dropout = float(self._args.get("b_dropout", b_dropout))

        # EWMA coefficients (keep-old weights)
        self.stat_alpha = float(self._args.get("stat_alpha", stat_alpha))
        self.sys_alpha = float(self._args.get("sys_alpha", sys_alpha))
        self.avail_alpha = float(self._args.get("avail_alpha", avail_alpha))

        # Preferred round duration (global) — backward compat, unused in data-only
        self.T: float = float(self._args.get("init_T", init_T))
        self.min_time = float(self._args.get("min_time", min_time))

        # Client-side parameters for OptimizationAtClient — backward compat
        self.beta = float(self._args.get("beta", beta))
        self.I_fix = int(self._args.get("I_fix", I_fix))

        # ── Data-only mode configuration ────────────────────────────────────
        self.data_only = self._as_bool(
            self._args.get("pyramidfl_data_only", True)
        )
        self.explore_fraction = float(
            self._args.get("pyramidfl_explore_fraction", self.epsilon)
        )
        self.stat_history_len = int(
            self._args.get("pyramidfl_stat_history_len", 10)
        )
        self.utility_epsilon = float(
            self._args.get("pyramidfl_utility_epsilon", 1e-12)
        )
        self.use_label_diversity = self._as_bool(
            self._args.get("pyramidfl_use_label_diversity", False)
        )
        self.label_diversity_weight = float(
            self._args.get("pyramidfl_label_diversity_weight", 0.0)
        )

        # In data-only mode, system utility should NOT drive explore
        self.use_system_utility_in_explore = self._as_bool(
            self._args.get("pyramidfl_use_system_utility_in_explore", False)
        )
        self.force_one_explore_when_k2 = self._as_bool(
            self._args.get("pyramidfl_force_one_explore_when_k2", True)
        )

        # Per-client state storage
        self._state: Dict[str, _ClientState] = {}
        
        # Local parameters to distribute to selected clients
        self._client_local_params: Dict[str, Dict[str, Any]] = {}

        # Debug/introspection
        self._last_R_stat: List[str] = []   # Last statistical utility ranking
        
        # Random seed initialized by base class call to with_random_seed if args present
        # but we also have local rng
        self._rng = random.Random(2024)
        if args:
            self.with_random_seed(args.random_seed)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def with_random_seed(self, seed: int = -1):
        super().with_random_seed(seed)
        # Also seed our local RNG
        # Note: super().with_random_seed uses 'random' module global seed.
        if seed > 0:
            self._rng = random.Random(seed)
        return self

    def _flatten_data(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flatten nested dictionary so that both prefixed and unprefixed
        leaf keys are available.

        Example input:
            {"train_record": {"avg_loss": 0.5, "num_samples_sum": 100}}
        Output includes:
            {"train_record_avg_loss": 0.5, "avg_loss": 0.5,
             "train_record_num_samples_sum": 100, "num_samples_sum": 100}

        Direct top-level keys are never overwritten by nested leaf keys.
        """
        result: dict = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # Recursively flatten nested dicts
                for sub_k, sub_v in self._flatten_data(v, new_key, sep=sep).items():
                    if sub_k not in result:
                        result[sub_k] = sub_v
                # Also merge the nested dict itself under its prefixed key
                if new_key not in result:
                    result[new_key] = v
            else:
                # Leaf: always add prefixed key
                result[new_key] = v
                # Also add raw key if it does not already exist
                if k not in result:
                    result[k] = v
                # Also add parent-prefixed key for direct access patterns
                if parent_key and new_key not in result:
                    result[new_key] = v
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Robust extraction helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_first_float(row: pd.Series, keys: List[str], default: float = 0.0) -> float:
        """Return the first valid float value from ``row`` for the given *keys*.

        Safely handles missing keys, ``None``, non-numeric values, and pandas NaN.
        """
        for key in keys:
            if key not in row:
                continue
            v = row[key]
            if v is None:
                continue
            try:
                fv = float(v)
                if pd.isna(fv):
                    continue
                return fv
            except (TypeError, ValueError):
                continue
        return default

    @staticmethod
    def _get_first_dict(row: pd.Series, keys: List[str]) -> Optional[Dict[str, float]]:
        """Return the first valid dict from ``row`` for the given *keys*.

        Returns ``None`` if none of the keys hold a non-empty dict.
        """
        for key in keys:
            if key not in row:
                continue
            v = row[key]
            if isinstance(v, dict) and len(v) > 0:
                try:
                    return {str(k): float(cnt) for k, cnt in v.items()}
                except (TypeError, ValueError):
                    pass
        return None

    def select(self, client_list: list, select_number: int = -1):
        """
        Main entry point for framework selection
        """
        if select_number <= 0:
            select_number = self.select_number # Use property from base class

        # Prepare metric matrix by flattening client data
        metric_matrix = {}
        for cid, data in self._clients_data_dict.items():
            if isinstance(data, dict):
                metric_matrix[cid] = self._flatten_data(data)
            else:
                metric_matrix[cid] = data

        name_dict = {str(c.node_id): str(getattr(c, "name", c.node_id)) for c in client_list}
        
        selected_ids = self.select_clients_internal(
            metric_matrix=metric_matrix,
            name_dict=name_dict,
            number=select_number,
            current_round=self.select_round
        )
        
        return [c for c in client_list if str(c.node_id) in selected_ids]

    def select_clients_internal(
        self,
        metric_matrix: Any,
        name_dict: Dict[str, str],
        number: int,
        break_client_set: Optional[Iterable[str]] = None,
        current_round: int = 0
    ) -> List[str]:
        """
        Server-side client selection — data-only PyramidFL-inspired pipeline.
        """
        if number <= 0:
            self._print_selected_clients([], name_dict)
            return []
        
        # Process break client set
        break_set = {str(x) for x in (break_client_set or [])}

        # 0. Normalize metrics to DataFrame and filter invalid clients
        df = self._to_dataframe(metric_matrix, name_dict)

        # Ensure client_id column exists and filter broken clients
        if not df.empty:
            if "client_id" not in df.columns:
                df["client_id"] = list(name_dict.keys())[: len(df)]
            df["client_id"] = df["client_id"].astype(str)
            df = df[~df["client_id"].isin(break_set)].copy()

        # C: All available clients at server
        C: List[str] = list(name_dict.keys())
        C = [cid for cid in C if cid not in break_set]

        # ------------------------------------------------------------------
        # GetClientFeedback  (data-only)
        # ------------------------------------------------------------------
        F_stat, F_sys, sample_num_map, shard_keep_ratio_map = self._get_client_feedback(df)

        # ------------------------------------------------------------------
        # UpdateClient  (data-only — no T filtering)
        # ------------------------------------------------------------------
        C_E, Util_stat, t_map = self._update_client(
            C, F_stat, F_sys, sample_num_map, shard_keep_ratio_map, current_round
        )

        # In data-only mode C_E is never empty (all clients are eligible)
        if not C_E:
            C_E = C
            Util_stat = {cid: self._state.get(cid, _ClientState()).util_stat_ema for cid in C_E}
            t_map = {cid: 1.0 for cid in C_E}

        # ------------------------------------------------------------------
        # UpdatePreferDuration  (no-op in data-only mode)
        # ------------------------------------------------------------------
        self.T = self._update_preferred_duration(F_stat, F_sys, self.T, self.delta_T)

        # ------------------------------------------------------------------
        # Calculate global utility  (data-only with anti-collapse bonuses)
        # ------------------------------------------------------------------
        Util = self._compute_global_utility(
            C_E, Util_stat, t_map, sample_num_map, shard_keep_ratio_map, current_round
        )

        # ------------------------------------------------------------------
        # Exploit / Explore split
        # ------------------------------------------------------------------
        K = min(number, len(C_E))
        if K <= 1:
            k_explore = 0
            k_exploit = K
        else:
            k_explore = max(1, int(round(self.explore_fraction * K)))
            if self.force_one_explore_when_k2 and K >= 2:
                k_explore = max(1, k_explore)
            k_explore = min(k_explore, K - 1)
            k_exploit = K - k_explore

        C_star = self._select_for_exploit(C_E, Util, k_exploit)

        # ------------------------------------------------------------------
        # SelectForExplore  (data-only — anti-starvation, no Util_sys)
        # ------------------------------------------------------------------
        C_opt = self._select_for_explore(C_E, C_star, k_explore, current_round)

        # ------------------------------------------------------------------
        # RankingClients(F_stat)
        # ------------------------------------------------------------------
        self._last_R_stat = self._ranking_clients(F_stat)

        # Fallback: random selection if nothing was selected
        if not C_opt and C_E:
            console.warning(
                f"[PyramidFL] No clients selected in round {current_round}. "
                f"Randomly selecting {min(number, len(C_E))} clients."
            )
            selection_size = min(number, len(C_E))
            selected_indices = self._rng.sample(range(len(C_E)), selection_size)
            C_opt = [C_E[i] for i in selected_indices]

        # ------------------------------------------------------------------
        # Deduplicate and truncate to select_number
        # ------------------------------------------------------------------
        seen: set = set()
        unique: List[str] = []
        for cid in C_opt:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)
        C_opt = unique[:number]

        # ------------------------------------------------------------------
        # Update participation_count and last_selected_round (ONCE, here)
        # ------------------------------------------------------------------
        for cid in C_opt:
            state = self._state.setdefault(cid, _ClientState())
            state.participation_count += 1
            state.last_selected_round = int(current_round)

        # ------------------------------------------------------------------
        # Generate client-side parameters  (data-only neutral)
        # ------------------------------------------------------------------
        if C_opt:
            self._generate_client_local_params(C_opt, t_map)

        # Print selection result and return
        self._print_selected_clients(C_opt, name_dict)
        return C_opt

    def _get_client_feedback(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Extract per-client data-only feedback from the metric DataFrame.

        Returns (F_stat, F_sys, sample_num_map, shard_keep_ratio_map).
        In data-only mode F_sys[cid] = 1.0 and shard_keep_ratio_map[cid] = 1.0
        for every client — system heterogeneity is NOT modelled.

        Statistical utility formula:
            F_stat[cid] = sqrt(max(n_i, 1.0)) * max(lp2, 0.0)
        where lp2 is the best-available loss/statistic value and n_i is the
        sample count.  Falls back to sqrt(n_i) * utility_epsilon when all
        loss values are missing or zero.
        """
        # Loss / statistical utility keys in priority order
        _LOSS_KEYS = [
            "sqrt_train_loss_power_two_sum",
            "train_record_sqrt_train_loss_power_two_sum",
            "train_loss_sum",
            "train_record_train_loss_sum",
            "avg_loss",
            "train_record_avg_loss",
            "loss",
            "train_record_loss",
            "batch_loss",
            "train_record_batch_loss",
        ]
        # Sample count keys in priority order
        _SAMPLE_KEYS = [
            "num_samples_sum",
            "train_record_num_samples_sum",
            "data_sample_num",
            "train_record_data_sample_num",
            "sample_num",
            "n_data",
            "num_samples",
        ]

        F_stat: Dict[str, float] = {}
        F_sys: Dict[str, float] = {}
        sample_num_map: Dict[str, float] = {}
        shard_keep_ratio_map: Dict[str, float] = {}

        for _, row in df.iterrows():
            cid = str(row["client_id"])

            # System feedback: neutral in data-only mode
            F_sys[cid] = 1.0
            shard_keep_ratio_map[cid] = 1.0

            # Extract sample count
            n_i = self._get_first_float(row, _SAMPLE_KEYS, 1.0)
            n_i = max(n_i, 1.0)
            sample_num_map[cid] = float(n_i)

            # Extract loss/statistic value
            lp2 = self._get_first_float(row, _LOSS_KEYS, 0.0)
            lp2 = max(lp2, 0.0)

            if lp2 > 0.0:
                F_stat[cid] = math.sqrt(n_i) * lp2
            else:
                # All-zero or missing loss → fallback to avoid selection collapse
                F_stat[cid] = math.sqrt(n_i) * self.utility_epsilon

        return F_stat, F_sys, sample_num_map, shard_keep_ratio_map

    def _update_client(
        self,
        C: List[str],
        F_stat: Dict[str, float],
        F_sys: Dict[str, float],
        sample_num_map: Dict[str, float],
        shard_keep_ratio_map: Dict[str, float],
        current_round: int,
    ) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
        """
        Update per-client state with fresh feedback (data-only mode).

        In data-only mode every client is always eligible (no T filtering).
        For clients with fresh F_stat the EMA and stat_history are updated;
        clients without fresh feedback keep their previous util_stat_ema.
        """
        C_E: List[str] = []
        Util_stat: Dict[str, float] = {}
        t_map: Dict[str, float] = {}

        for cid in C:
            state = self._state.setdefault(cid, _ClientState())

            if cid in F_stat:
                # Client has fresh statistical feedback
                s_i = float(F_stat.get(cid, 0.0))

                # Update statistical utility EWMA
                state.util_stat_ema = (
                    self.stat_alpha * state.util_stat_ema
                    + (1.0 - self.stat_alpha) * s_i
                )

                state.stat_history.append(s_i)
                if len(state.stat_history) > self.stat_history_len:
                    state.stat_history.pop(0)

                if state.stat_history:
                    state.stat_mean = float(np.mean(state.stat_history))
                    state.stat_std = float(np.std(state.stat_history))
                else:
                    state.stat_mean = 0.0
                    state.stat_std = 0.0

                state.last_sample_num = float(sample_num_map.get(cid, state.last_sample_num))
                state.last_shard_keep = float(shard_keep_ratio_map.get(cid, state.last_shard_keep))
                state.last_seen_round = current_round

            # In data-only mode every client is always eligible
            C_E.append(cid)
            Util_stat[cid] = state.util_stat_ema
            t_map[cid] = 1.0

        return C_E, Util_stat, t_map

    def _update_preferred_duration(
        self,
        F_stat: Dict[str, float],
        F_sys: Dict[str, float],
        T: float,
        delta_T: float,
    ) -> float:
        """
        No-op in data-only mode: preferred duration is meaningless for simulation.
        Returns T unchanged.
        """
        if self.data_only:
            return T

        # Original paper logic (kept for backward compatibility)
        if not F_sys:
            return T
        stat_sum = sum(F_stat.values()) if F_stat else 0.0
        if stat_sum > 0.0:
            new_T = T + delta_T
        else:
            new_T = T - delta_T
        return max(self.min_time, new_T)

    def _compute_global_utility(
        self,
        C_E: List[str],
        Util_stat: Dict[str, float],
        t_map: Dict[str, float],
        sample_num_map: Dict[str, float],
        shard_keep_ratio_map: Dict[str, float],
        current_round: int = 0,
    ) -> Dict[str, float]:
        """
        Compute data-only global utility with anti-collapse bonuses.

        base = max(Util_stat[cid], 0.0)

        Two small epsilon-scaled bonuses are added:
          • uncertainty_bonus  – favours clients with less stat history
          • staleness_bonus    – favours clients not selected recently

        Does NOT use time, T, system utility, or shard keep ratio.
        """
        Util: Dict[str, float] = {}

        for cid in C_E:
            state = self._state.setdefault(cid, _ClientState())
            base = max(Util_stat.get(cid, state.util_stat_ema), 0.0)

            # Uncertainty bonus: less history → higher bonus
            history_len = len(state.stat_history)
            uncertainty_bonus = 1.0 / math.sqrt(history_len + 1.0)

            # Staleness bonus: not selected recently → higher bonus
            staleness_bonus = 0.0
            if state.last_selected_round >= 0:
                staleness_bonus = min(
                    1.0,
                    max(0, current_round - state.last_selected_round) / 10.0,
                )
            else:
                staleness_bonus = 1.0

            Util[cid] = (
                base
                + self.utility_epsilon * uncertainty_bonus
                + self.utility_epsilon * staleness_bonus
            )

        return Util

    def _select_for_exploit(
        self,
        C_E: List[str],
        Util: Dict[str, float],
        k_exploit: int,
    ) -> List[str]:
        """
        Select top k_exploit clients by data utility (exploitation).

        Does NOT update participation_count — that is done once in
        ``select_clients_internal`` after the final C_opt is produced.
        """
        if not C_E or k_exploit <= 0:
            return []

        k_exploit = min(k_exploit, len(C_E))

        sorted_by_util = sorted(
            C_E, key=lambda cid: Util.get(cid, 0.0), reverse=True
        )
        return sorted_by_util[:k_exploit]

    def _select_for_explore(
        self,
        C_E: List[str],
        C_star: List[str],
        k_explore: int,
        current_round: int,
    ) -> List[str]:
        """
        Select k_explore clients for exploration (data-only mode).

        Chooses from C_E \\ C_star using anti-starvation and uncertainty
        priorities.  Does NOT use Util_sys, last_time, or T.

        Sort key (ascending priority):
          1. clients selected least recently (last_selected_round)
          2. clients with lower participation_count
          3. clients with fewer stat history records
          4. optionally higher statistical utility (negative → earlier)
          5. random tie-breaker

        Does NOT update participation_count — that is done once in
        ``select_clients_internal`` after the final C_opt is produced.
        """
        if k_explore <= 0:
            return list(C_star)

        remaining = [cid for cid in C_E if cid not in C_star]
        if not remaining:
            return list(C_star)

        k_explore = min(k_explore, len(remaining))

        remaining_sorted = sorted(
            remaining,
            key=lambda cid: (
                int(self._state.get(cid, _ClientState()).last_selected_round),
                int(self._state.get(cid, _ClientState()).participation_count),
                len(self._state.get(cid, _ClientState()).stat_history),
                -float(self._state.get(cid, _ClientState()).util_stat_ema),
                self._rng.random(),
            ),
        )
        C_explore = remaining_sorted[:k_explore]

        return list(C_star) + C_explore

    def _ranking_clients(self, F_stat: Dict[str, float]) -> List[str]:
        """
        Algorithm 1: RankingClients(F_stat)
        """
        return sorted(F_stat.keys(), key=lambda cid: F_stat[cid], reverse=True)

    def _generate_client_local_params(self, C_opt: List[str], t_map: Dict[str, float]) -> None:
        """
        Generate client-side parameters (data-only neutral).

        In this simulation environment real shard keep ratio and adaptive
        local iterations are not executed.  Returns neutral params:
          shard_keep_ratio = 1.0
          adaptive_iter    = I_fix
        """
        self._client_local_params = {
            cid: {
                "shard_keep_ratio": 1.0,
                "adaptive_iter": self.I_fix,
            }
            for cid in C_opt
        }

    def _extract_time(self, row: pd.Series) -> float:
        """
        Extract wall-clock time cost from metric row.
        Priority: latency (from runner) > time_cost > other keys
        """
        # First try latency which is added by the runner
        if "latency" in row and row["latency"] is not None:
            try:
                return max(self.min_time, float(row["latency"]))
            except (TypeError, ValueError):
                pass
        
        # Then try other time-related keys
        time_keys = [
            "time_cost", "total_time_cost", "round_time",
            "training_time", "train_time"
        ]
        for key in time_keys:
            if key in row and row[key] is not None:
                try:
                    return max(self.min_time, float(row[key]))
                except (TypeError, ValueError):
                    continue
        return self.min_time

    def _to_dataframe(self, metric_matrix: Any, name_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Normalize various metric matrix formats to a pandas DataFrame.
        """
        if isinstance(metric_matrix, pd.DataFrame):
            df = metric_matrix.copy()
        elif isinstance(metric_matrix, list):
            df = pd.DataFrame(metric_matrix) if metric_matrix else pd.DataFrame()
        elif isinstance(metric_matrix, dict):
            records = [
                {**v, "client_id": cid} if isinstance(v, dict) else {"client_id": cid, "value": v}
                for cid, v in metric_matrix.items()
            ]
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame()

        if "client_id" not in df.columns and name_dict:
            df["client_id"] = list(name_dict.keys())[: len(df)]
            
        return df

    def _print_selected_clients(self, selected: List[str], name_dict: Dict[str, str]) -> None:
        """
        Print selected clients (debug/introspection).
        """
        pass # Suppressed output to avoid console spam

