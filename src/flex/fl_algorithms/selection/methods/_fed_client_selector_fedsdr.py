from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_FedSDR(FedClientSelector):
    """
    FedSDR: Federated Dynamic Client Selection for Fairness Guarantee in
    Heterogeneous Edge Computing.
    (Ying-Chi Mao et al. – Journal of Computer Science and Technology, 2024)

    ── Algorithm goal ────────────────────────────────────────────────────────
    Simultaneously reduce two fairness metrics:
      Var(Fre) = (1/n) Σ (Fre_i − mean(Fre))²   # participation fairness
      Var(Acc) = (1/n) Σ (Acc_i − mean(Acc))²   # local performance fairness

    ── Three-module pipeline (per round) ────────────────────────────────────
    1. UPDATE  – update local computational efficiency e_i for participated
                 clients:  e_i ← |d_i| / t_i^latest
                 (initial value when no prior training: governed by
                  fedsdr_efficiency_init)

    2. GROUP   – sort clients by e_i and partition into m contiguous
                 efficiency tiers, where m is derived from select_number
                 (m = ceil(select_number / 2)).  Clients with similar
                 computational efficiency stay in the same group, so every
                 efficiency tier gets a chance to contribute.

    3. DYNAMIC-SELECT – inside each group k:
         a. Compute per-client balance degree
                b_{i,k} = exp( −D_KL( A_{i,k} ‖ U ) )
            where A_{i,k} is the local label distribution and U is the
            uniform distribution over the **global** class universe.
            If label distribution is unavailable the behaviour is governed
            by fedsdr_missing_label_policy:
              • "neutral"     → return 0.5 (default, most conservative)
              • "loss_proxy"  → use exp(−avg_loss_i) as an engineering
                                 fallback (NOT paper-faithful)
              • "raise"       → raise ValueError
         b. Sort clients by b ascending (most skewed → most balanced).
         c. FedSDR-style representative selection:
              pick the client with the **smallest** b (most skewed /
              most extreme data distribution) and the client with the
              **largest** b (most balanced / most ordinary distribution).
            This ensures the model sees both typical and edge distributions
            from every efficiency tier.

    Total clients selected per round: 2 per group × m groups,
    truncated to select_number.

    ── Configuration (yaml) ──────────────────────────────────────────────────
      fedsdr_num_classes          : int    strongly recommended for datasets
                                           such as MNIST/FMNIST/CIFAR10/CINIC10
                                           (e.g. 10).  If omitted the union of
                                           all labels seen across clients is used.
      fedsdr_missing_label_policy : str    "neutral" (default), "loss_proxy",
                                           or "raise".
      fedsdr_efficiency_init      : str    "sample_count" (default) or "constant".
      fedsdr_group_update_period  : int    how many rounds between re-groupings
                                           (default 1)
      fedsdr_eps                  : float  small constant ε to avoid o=0
                                           (default 1e-6, kept for compatibility)
    """

    # Keys tried (in order) when reading per-round training time from train_record
    _TIME_KEYS: Tuple[str, ...] = (
        "time_cost", "training_time", "train_time", "duration", "round_time",
        "total_time_cost",
    )

    def __init__(self, args: FedClientSelectorArgs | None = None):
        super().__init__(args)
        self._args.select_method = "fedsdr"

        # ── new FedSDR-paper-aligned configuration ─────────────────────────
        self._num_classes: Optional[int] = None
        raw_nc = self._args.get("fedsdr_num_classes", None)
        if raw_nc is not None:
            self._num_classes = int(raw_nc)

        self._missing_label_policy: str = str(
            self._args.get("fedsdr_missing_label_policy", "neutral")
        )
        if self._missing_label_policy not in ("neutral", "loss_proxy", "raise"):
            raise ValueError(
                f"fedsdr_missing_label_policy must be 'neutral', 'loss_proxy', "
                f"or 'raise', got '{self._missing_label_policy}'"
            )

        self._efficiency_init: str = str(
            self._args.get("fedsdr_efficiency_init", "sample_count")
        )
        if self._efficiency_init not in ("sample_count", "constant"):
            raise ValueError(
                f"fedsdr_efficiency_init must be 'sample_count' or 'constant', "
                f"got '{self._efficiency_init}'"
            )

        self._group_update_period: int = int(self._args.get("fedsdr_group_update_period", 1))
        self._eps: float = float(self._args.get("fedsdr_eps", 1e-6))

        # Per-client persistent state
        # { client_id: { "e": float,          # computational efficiency
        #                "num_samples": float, # |d_i|
        #                "has_time": bool,    # whether a real timing-based e_i has been observed
        # }}
        self._cstate: Dict[str, Dict] = {}

        # Current grouping: {group_idx: [client_id, ...]}
        self._groups: Dict[int, List[str]] = {}
        self._round_idx: int = 0

        # Cached global class keys for KL computation (populated lazily)
        self._class_keys: Optional[List[str]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Data extraction helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_record(self, cid: str) -> Optional[Dict]:
        entry = (self._clients_data_dict.get(cid)
                 or self._clients_data_dict.get(str(cid)))
        if not isinstance(entry, dict):
            return None
        record = entry.get("train_record", entry)
        return record if isinstance(record, dict) else None

    def _extract_num_samples(self, record: Dict) -> Optional[float]:
        for key in ("num_samples_sum", "data_sample_num", "num_samples"):
            v = record.get(key)
            if v is not None:
                try:
                    return max(float(v), 1.0)
                except (TypeError, ValueError):
                    pass
        return None

    def _extract_train_time(self, record: Dict) -> Optional[float]:
        for key in self._TIME_KEYS:
            v = record.get(key)
            if v is not None:
                try:
                    t = float(v)
                    if t > 0:
                        return t
                except (TypeError, ValueError):
                    pass
        return None

    def _extract_avg_loss(self, record: Dict) -> Optional[float]:
        for key in ("avg_loss", "train_loss_sum", "sqrt_train_loss_power_two_sum", "loss"):
            v = record.get(key)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    def _extract_label_dist(self, record: Dict) -> Optional[Dict[str, float]]:
        """Return {class: count} if the train_record carries label distribution."""
        for key in ("label_dist", "label_counts", "class_distribution", "label_distribution"):
            v = record.get(key)
            if isinstance(v, dict) and len(v) > 0:
                try:
                    return {str(k): float(cnt) for k, cnt in v.items()}
                except (TypeError, ValueError):
                    pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 1 – UPDATE: refresh computational efficiency
    # ─────────────────────────────────────────────────────────────────────────

    def _update_efficiency(self, all_cids: List[str]) -> None:
        """Update e_i for every client that has fresh train_record data.

        When latest training time is available:
            e_i = num_samples / train_time   (paper §4.1)

        When training time is unavailable and the client has never had a
        real timing observation (has_time == False):
            • fedsdr_efficiency_init == "sample_count" → e_i = num_samples
            • fedsdr_efficiency_init == "constant"     → e_i = 1.0

        Once a client has observed a real training time (has_time becomes True),
        the efficiency from that observation is kept until another observation
        arrives.
        """
        for cid in all_cids:
            if cid not in self._cstate:
                self._cstate[cid] = {"e": 1.0, "num_samples": 1.0, "has_time": False}

            record = self._get_record(cid)
            if record is None:
                continue

            num_samples = self._extract_num_samples(record)
            if num_samples is not None:
                self._cstate[cid]["num_samples"] = num_samples

            train_time = self._extract_train_time(record)

            if train_time is not None:
                # e_i = |d_i| / t_i  (paper §4.1)
                n = self._cstate[cid]["num_samples"]
                self._cstate[cid]["e"] = n / train_time
                self._cstate[cid]["has_time"] = True
            elif not self._cstate[cid].get("has_time", False):
                # No timing observation yet — use the configured init strategy
                if self._efficiency_init == "sample_count":
                    self._cstate[cid]["e"] = self._cstate[cid]["num_samples"]
                else:
                    self._cstate[cid]["e"] = 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 2 – GROUP: partition clients by efficiency into m groups
    # ─────────────────────────────────────────────────────────────────────────

    def _build_groups(self, all_cids: List[str], m: int) -> Dict[int, List[str]]:
        """
        Sort clients by e_i descending then split into m **contiguous**
        efficiency tiers.  Clients with similar computational efficiency
        stay in the same group (paper §5).

        Uses:  k = min(i * m // n, m - 1)
        rather than modulo assignment, so that the sorted list is partitioned
        into consecutive blocks.
        """
        n = len(all_cids)
        sorted_cids = sorted(
            all_cids,
            key=lambda cid: self._cstate.get(cid, {}).get("e", 1.0),
            reverse=True,
        )
        groups: Dict[int, List[str]] = {k: [] for k in range(m)}
        for i, cid in enumerate(sorted_cids):
            k = min(i * m // n, m - 1)
            groups[k].append(cid)
        return groups

    # ─────────────────────────────────────────────────────────────────────────
    # Global class-key helper (for KL computation over full class universe)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_class_keys(self) -> List[str]:
        """
        Return the ordered list of class keys over which KL divergence is
        computed.

        If ``fedsdr_num_classes`` was provided, returns
        ``["0", "1", ..., str(num_classes-1)]``.

        Otherwise scans all available label distributions across clients
        and returns the sorted union of all label keys seen.  This fallback
        is less robust because a client that only has label "7" may appear
        perfectly balanced if the universe is inferred as {"7"} alone.
        """
        if self._class_keys is not None:
            return self._class_keys

        if self._num_classes is not None:
            self._class_keys = [str(i) for i in range(self._num_classes)]
            return self._class_keys

        # Infer from data — collect all label keys ever reported
        all_keys: set = set()
        for cid in self._cstate:
            record = self._get_record(cid)
            if record is not None:
                ld = self._extract_label_dist(record)
                if ld is not None:
                    all_keys.update(ld.keys())
        self._class_keys = sorted(all_keys, key=lambda k: (k.isdigit(), k))
        return self._class_keys

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 3 – DYNAMIC-SELECT: representativity-based selection inside group
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _kl_from_uniform(label_dist: Dict[str, float],
                          class_keys: List[str]) -> float:
        """
        D_KL( A ‖ U ) where U is the discrete uniform distribution over
        the **global class universe** given by ``class_keys``.

        Classes with zero count in ``label_dist`` contribute zero to the
        KL sum (0 * log(0 / uniform_p) ≡ 0).  A client holding only one
        label (e.g. {"7": 6000}) is correctly identified as highly skewed
        because the zero-count classes widen the divergence from uniform.
        """
        if not class_keys:
            return 0.0
        n_classes = len(class_keys)
        if n_classes <= 1:
            return 0.0

        total = sum(label_dist.get(k, 0.0) for k in class_keys)
        if total <= 0:
            return 0.0

        uniform_p = 1.0 / n_classes
        kl = 0.0
        for k in class_keys:
            c = label_dist.get(k, 0.0)
            if c <= 0:
                continue
            p = c / total
            kl += p * math.log(p / uniform_p)
        return max(kl, 0.0)

    def _balance_degree(self, cid: str) -> float:
        """
        b_i = exp( −D_KL( A_i ‖ U ) )  ∈ (0, 1]          (paper §6.2)

        KL is computed over the **global** class universe (see
        ``_get_class_keys``).

        When label distribution is unavailable the behaviour depends on
        ``fedsdr_missing_label_policy``:
          • ``"neutral"``    → return 0.5 (default)
          • ``"loss_proxy"`` → return exp(−max(avg_loss, 0.0))
                                (**engineering fallback, not paper-faithful**)
          • ``"raise"``      → raise ValueError
        """
        record = self._get_record(cid)
        if record is not None:
            label_dist = self._extract_label_dist(record)
            if label_dist is not None:
                class_keys = self._get_class_keys()
                kl = self._kl_from_uniform(label_dist, class_keys)
                return math.exp(-kl)

        # Label distribution unavailable — follow configured policy
        policy = self._missing_label_policy
        if policy == "loss_proxy":
            if record is not None:
                avg_loss = self._extract_avg_loss(record)
                if avg_loss is not None:
                    return math.exp(-max(avg_loss, 0.0))
            return 0.5  # fallback when even loss is unavailable
        elif policy == "raise":
            raise ValueError(
                f"FedSDR: label distribution unavailable for client '{cid}' "
                f"and fedsdr_missing_label_policy='raise'.  "
                f"Provide label_dist in train_record or change the policy."
            )
        else:
            # "neutral" — default
            return 0.5

    def _select_from_group(self, group_cids: List[str]) -> List[str]:
        """
        FedSDR-style representative selection from one efficiency group.

        Computes balance degree for each client, sorts by b ascending,
        then selects:
          • one client with the **smallest** b (most skewed / most extreme
            data distribution), and
          • one client with the **largest** b (most balanced / most ordinary
            data distribution).

        Returns at most 2 distinct clients.  If group size ≤ 2, returns
        all clients in the group.
        """
        if len(group_cids) <= 2:
            return list(group_cids)

        # Compute balance degrees and sort ascending (most skewed first)
        b_values = [(cid, self._balance_degree(cid)) for cid in group_cids]
        b_values.sort(key=lambda x: x[1])

        most_skewed_cid = b_values[0][0]       # smallest b
        most_balanced_cid = b_values[-1][0]     # largest b

        if most_skewed_cid == most_balanced_cid:
            return [most_skewed_cid]
        return [most_skewed_cid, most_balanced_cid]

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry-point
    # ─────────────────────────────────────────────────────────────────────────

    def select(self, client_list: list, select_number: int = -1) -> list:
        """
        Execute one round of FedSDR:
          UPDATE → GROUP → DYNAMIC-SELECT → return up to select_number clients.

        m = ceil(select_number / 2) groups; 2 clients selected per group;
        final list truncated to select_number after deduplication.
        """
        if select_number <= 0:
            select_number = self._args.select_number

        m = max(1, math.ceil(select_number / 2))
        self._round_idx += 1

        all_cids = [str(c.node_id) for c in client_list]
        cid_to_client = {str(c.node_id): c for c in client_list}

        # MODULE 1: UPDATE efficiency estimates
        self._update_efficiency(all_cids)

        # MODULE 2: GROUP (rebuild according to update period)
        if self._round_idx == 1 or (self._round_idx % self._group_update_period == 0):
            self._groups = self._build_groups(all_cids, m)

        # MODULE 3: DYNAMIC-SELECT – pick extreme pair per group
        selected_ids: List[str] = []
        for k in range(m):
            group_cids = self._groups.get(k, [])
            # Keep only clients that are currently available
            group_cids = [cid for cid in group_cids if cid in cid_to_client]
            if not group_cids:
                continue
            selected_ids.extend(self._select_from_group(group_cids))

        # Deduplicate while preserving order
        seen: set = set()
        unique_ids: List[str] = []
        for cid in selected_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        # Truncate to select_number
        unique_ids = unique_ids[:select_number]

        return [cid_to_client[cid] for cid in unique_ids if cid in cid_to_client]
