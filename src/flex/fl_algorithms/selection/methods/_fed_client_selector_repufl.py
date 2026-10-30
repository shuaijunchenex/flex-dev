from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_RepuFL(FedClientSelector):
    """
    RepuFL: Reputation-Aware Client Selection in Federated Learning for Mobile Networks.
    (Subhash Sagar, Adnan Mahmood – IEEE ICWS 2025, work-in-progress)

    ── Algorithm goal ────────────────────────────────────────────────────────
    Rather than selecting clients purely by instant utility (e.g. loss magnitude),
    RepuFL maintains a dynamic, cross-round *reputation score* R_i for every
    client i that quantifies its long-term credibility.  Clients with higher
    reputation are preferred in each round's selection.

    ── Four reputation factors (from paper public abstract) ──────────────────
    The reputation of client i at round r is determined by four adaptive,
    dynamically updated factors:

      A_i^(r)  – Historical Accuracy
                 Reflects how reliably client i has contributed useful updates
                 over past rounds.  Clients with consistently low loss (high
                 accuracy proxy) accumulate a higher A score.
                 [Engineering impl: EWMA of (1 − normalised_loss)]

      Q_i^(r)  – Data Quality
                 Reflects the reliability and richness of the client's local
                 dataset.  Clients with more samples are treated as a proxy for
                 richer, less noisy data.
                 [Engineering impl: min-max normalised sample count per round]

      W_i^(r)  – Willingness
                 Reflects whether the client is willing and able to participate.
                 Clients that consistently respond and submit updates score
                 higher; clients that drop out or miss rounds score lower.
                 [Engineering impl: EWMA of binary participation signal (1/0)]

      C_i^(r)  – Behavioral Consistency
                 Reflects stability of the client's behaviour across rounds.
                 Clients whose loss fluctuates unpredictably are penalised;
                 clients with stable training trajectories are rewarded.
                 [Engineering impl: 1 / (1 + variance_of_loss_history)]

    ── Reputation fusion (paper notation) ───────────────────────────────────
      R_i^(r) = Φ( A_i^(r), Q_i^(r), W_i^(r), C_i^(r) )

    The precise form of Φ is not given in the public abstract.
    This implementation uses an equal-weighted linear combination:
      R_i = w_A·A_i + w_Q·Q_i + w_W·W_i + w_C·C_i
    where the weights default to 0.25 and are configurable via yaml:
      repufl_weight_accuracy, repufl_weight_quality,
      repufl_weight_willingness, repufl_weight_consistency

    ── Per-round algorithm flow ──────────────────────────────────────────────
    Before each round r:
      1. Ingest last round's training feedback (loss, sample count, participation).
      2. Update A_i, Q_i, W_i, C_i for every candidate client.
      3. Compute R_i = Φ(A_i, Q_i, W_i, C_i).
      4. Select the top-K clients by R_i → S^(r).

    After round r completes, the updated factors carry forward to round r+1,
    making RepuFL an inherently cumulative, cross-round selection mechanism.

    ── Important disclaimer ──────────────────────────────────────────────────
    The four factor names and their semantic roles are confirmed by the paper's
    public abstract.  The specific update equations, normalisation choices,
    EWMA coefficients, and the fusion function Φ are *engineering approximations*
    and should be updated once the full paper is available.
    """

    def __init__(self, args: FedClientSelectorArgs | None = None):
        super().__init__(args)
        self._args.select_method = "repufl"

        # EWMA decay factor (keep-old weight) for A and W
        self._ewma_alpha: float = float(self._args.get("repufl_ewma_alpha", 0.7))

        # Reputation factor weights (must sum to 1 for correct [0,1] range)
        self._w_A: float = float(self._args.get("repufl_weight_accuracy",    0.25))
        self._w_Q: float = float(self._args.get("repufl_weight_quality",     0.25))
        self._w_W: float = float(self._args.get("repufl_weight_willingness", 0.25))
        self._w_C: float = float(self._args.get("repufl_weight_consistency", 0.25))
        self._use_willingness: bool = self._as_bool(self._args.get("repufl_use_willingness", True))

        # K=2 in one-label non-IID benefits from one forced exploration slot.
        self._force_explore_slots: int = int(self._args.get("repufl_force_explore_slots", 1))
        self._force_explore_when_k2: bool = self._as_bool(
            self._args.get("repufl_force_explore_when_k2", True)
        )

        # Sliding window length for behavioral consistency (C factor).
        # Keeping only the last N rounds avoids old high-loss rounds
        # permanently penalising long-running participants.
        self._loss_history_window: int = int(self._args.get("repufl_loss_history_window", 10))

        seed = int(getattr(self._args, "random_seed", -1))
        self._rng = random.Random(seed if seed > 0 else 2024)

        # Per-client persistent state ─────────────────────────────────────────
        # {client_id: {
        #     "A": float,          # historical accuracy factor (EWMA)
        #     "Q": float,          # data quality factor
        #     "W": float,          # willingness factor (EWMA)
        #     "C": float,          # behavioral consistency factor
        #     "R": float,          # reputation score
        #     "loss_history": List[float],   # raw per-round loss values
        # }}
        self._state: Dict[str, Dict] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_state(self, client_id: str) -> Dict:
        if client_id not in self._state:
            self._state[client_id] = {
                "A": 0.5, "Q": 0.5, "W": 0.5, "C": 0.5, "R": 0.5,
                "loss_history": [],
                "selected_count": 0,
                "last_selected_round": -1,
                "last_seen_round": -1,
            }
        return self._state[client_id]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _extract_record(self, client_id) -> Optional[Dict]:
        """Return the train_record dict for this client, or None."""
        entry = (self._clients_data_dict.get(client_id)
                 or self._clients_data_dict.get(str(client_id)))
        if not isinstance(entry, dict):
            return None
        record = entry.get("train_record", entry)
        return record if isinstance(record, dict) else None

    def _extract_loss(self, record: Dict) -> Optional[float]:
        for key in ("avg_loss", "sqrt_train_loss_power_two_sum",
                    "train_loss_sum", "loss"):
            val = record.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Reputation update  (called once per round, before select)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_reputation(self, all_client_ids: List[str]):
        """
        Update all four reputation factors for every known client and
        compute the combined reputation score R_i.

        all_client_ids: IDs of all candidate clients this round.
        """
        # Collect this round's observations ──────────────────────────────────
        round_losses: Dict[str, float] = {}
        round_samples: Dict[str, float] = {}

        for cid in all_client_ids:
            record = self._extract_record(cid)
            if record is None:
                continue
            loss = self._extract_loss(record)
            if loss is not None:
                round_losses[cid] = loss
            samples = record.get("num_samples_sum")
            if samples is not None:
                try:
                    round_samples[cid] = float(samples)
                except (TypeError, ValueError):
                    pass

        # Normalise loss to [0,1] using min-max across this round ─────────────
        if round_losses:
            min_loss = min(round_losses.values())
            max_loss = max(round_losses.values())
            loss_range = max(max_loss - min_loss, 1e-8)
            norm_loss = {cid: (v - min_loss) / loss_range for cid, v in round_losses.items()}
        else:
            norm_loss = {}

        # Normalise sample count to [0,1] ─────────────────────────────────────
        if round_samples:
            min_s = min(round_samples.values())
            max_s = max(round_samples.values())
            s_range = max(max_s - min_s, 1e-8)
            norm_samples = {cid: (v - min_s) / s_range for cid, v in round_samples.items()}
        else:
            norm_samples = {}

        # Update per-client factors ───────────────────────────────────────────
        for cid in all_client_ids:
            s = self._get_state(cid)
            participated = cid in round_losses   # client responded this round

            # ── Willingness (W): EWMA of binary participation signal ──────────
            s["W"] = self._ewma_alpha * s["W"] + (1.0 - self._ewma_alpha) * (1.0 if participated else 0.0)

            if participated:
                s["last_seen_round"] = int(self.select_round)
                # ── Historical Accuracy (A): EWMA of (1 - normalised_loss) ────
                acc_proxy = 1.0 - norm_loss.get(cid, 0.5)
                s["A"] = self._ewma_alpha * s["A"] + (1.0 - self._ewma_alpha) * acc_proxy

                # ── Data Quality (Q): normalised sample count ─────────────────
                s["Q"] = norm_samples.get(cid, s["Q"])

                # ── Behavioral Consistency (C): 1 / (1 + variance of losses) ──
                s["loss_history"].append(round_losses[cid])
                # Sliding window: keep only the last N rounds to avoid old
                # high-loss rounds permanently penalising early participants.
                if len(s["loss_history"]) > self._loss_history_window:
                    s["loss_history"] = s["loss_history"][-self._loss_history_window:]
                if len(s["loss_history"]) >= 2:
                    mean_h = sum(s["loss_history"]) / len(s["loss_history"])
                    var_h = sum((x - mean_h) ** 2 for x in s["loss_history"]) / len(s["loss_history"])
                    s["C"] = 1.0 / (1.0 + var_h)
                # else C stays at its initialised value (0.5)

            # ── Combined reputation ───────────────────────────────────────────
            wW = self._w_W if self._use_willingness else 0.0
            weight_sum = max(1e-8, self._w_A + self._w_Q + self._w_C + wW)
            s["R"] = ((self._w_A * s["A"]
                       + self._w_Q * s["Q"]
                       + wW * s["W"]
                       + self._w_C * s["C"]) / weight_sum)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry-point
    # ─────────────────────────────────────────────────────────────────────────

    def select(self, client_list: list, select_number: int = -1) -> list:
        """
        Select the top-K clients ranked by reputation score.
        Falls back to all clients when no metrics are available yet.
        """
        if select_number <= 0:
            select_number = self._args.select_number

        all_cids = [str(c.node_id) for c in client_list]

        # Step 1: update reputation based on last round's feedback
        self._update_reputation(all_cids)

        if not all_cids:
            return []

        # Step 2: rank by reputation
        sorted_clients = sorted(
            client_list,
            key=lambda c: self._get_state(str(c.node_id))["R"],
            reverse=True,
        )

        K = min(select_number, len(sorted_clients))
        if K <= 0:
            return []

        force_slots = min(max(0, self._force_explore_slots), max(0, K - 1))
        if self._force_explore_when_k2 and K >= 2:
            force_slots = max(force_slots, 1)
        k_top = K - force_slots

        selected = sorted_clients[:k_top]
        selected_ids = {str(c.node_id) for c in selected}

        if force_slots > 0:
            candidates = [c for c in client_list if str(c.node_id) not in selected_ids]
            explore_sorted = sorted(
                candidates,
                key=lambda c: (
                    int(self._get_state(str(c.node_id)).get("selected_count", 0)),
                    int(self._get_state(str(c.node_id)).get("last_selected_round", -1)),
                    -float(self._get_state(str(c.node_id)).get("R", 0.0)),
                    self._rng.random(),
                ),
            )
            selected.extend(explore_sorted[:force_slots])

        for c in selected:
            s = self._get_state(str(c.node_id))
            s["selected_count"] = int(s.get("selected_count", 0)) + 1
            s["last_selected_round"] = int(self.select_round)

        return selected
