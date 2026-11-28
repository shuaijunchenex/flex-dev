from __future__ import annotations

import pandas as pd
import numpy as np
import math
from typing import Any, Dict, Iterable, List, Optional, Set

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_FedGRA(FedClientSelector):
    """
    FedGRA client selection algorithm using Grey Relational Analysis.
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        # Track staleness (rounds since last participation) for fairness
        # Key: client_id, Value: staleness count
        self._staleness: Dict[str, int] = {}
    
        
        # Load args
        self.fedgra_mode = self._args.get("fedgra_mode", "all_participate")
        self.min_participate_round = self._args.get("min_participate_round", 10) # staleness normalisation cap
        self.fedgra_eps = float(self._args.get("fedgra_eps", 0.5))
        # Optional metric weights, defaults to equal weighting
        self.fedgra_metric_weights = self._args.get("fedgra_metric_weights", None)
        self.fedgra_use_ewm = bool(self._args.get("fedgra_use_ewm", True))
        default_profile = "standard" if getattr(self._args, "select_method", "") == "fedgra_standard" else "keras"
        self.fedgra_metric_profile = self._args.get("fedgra_metric_profile", default_profile)

    def select(self, client_list: list, select_number: int = -1):
        """
        Main framework entry point
        """
        if select_number <= 0:
            select_number = self.select_number

        # 1. Update staleness for all available clients
        # Increment staleness for everyone present in this round's candidate list
        current_cids = [str(c.node_id) for c in client_list]
        for cid in current_cids:
            self._staleness[cid] = self._staleness.get(cid, 0) + 1

        # 2. Build GRG Matrix & Fairness Matrix
        # Try to compute GRG internally; fallback to provided grg/fedgra_score/loss.
        computed_grg = self._compute_gra_scores(current_cids)
        grg_matrix = {}

        for cid in current_cids:
            if computed_grg and cid in computed_grg:
                grg_matrix[cid] = computed_grg[cid]
                continue

            data = self._clients_data_dict.get(cid, {})
            val = 0.0
            if isinstance(data, dict):
                if "grg" in data:
                    val = float(data["grg"])
                elif "fedgra_score" in data:
                    val = float(data["fedgra_score"])
                else:
                    # Priority: use trainer metrics (avg_loss, train_loss_sum, etc.)
                    # These are now at top level after server extracts from nested train_record
                    val = float(data.get("avg_loss", data.get("train_loss_sum", data.get("loss", 0.0))))
            else:
                try:
                    val = float(data)
                except Exception:
                    val = 0.0

            grg_matrix[cid] = val

        # Fairness matrix is actually the staleness map
        fairness_matrix = {cid: self._staleness.get(cid, 0) for cid in current_cids}

        name_dict = {str(c.node_id): str(getattr(c, "name", c.node_id)) for c in client_list}

        # 3. Call the core algorithm
        selected_ids = self.select_clients(
            grg_matrix=grg_matrix,
            fairness_matrix=fairness_matrix,
            name_dict=name_dict,
            number=select_number,
            fedgra_mode=self.fedgra_mode,
            min_participate_round=self.min_participate_round
        )

        # 4. Post-selection: Reset staleness for selected clients
        for cid in selected_ids:
            self._staleness[cid] = 0

        # Return objects
        return [c for c in client_list if str(c.node_id) in selected_ids]

    def select_clients(self, grg_matrix, fairness_matrix, name_dict, number, 
                      fedgra_mode='all_participate', min_participate_round=10, 
                      break_client_set=None, **kwargs):
        """
        Select clients using FedGRA algorithm.
        
        Args:
            grg_matrix: Grey relational grade matrix (Client ID -> GRG Score)
            fairness_matrix: Client fairness scores (Client ID -> Staleness/Rounds)
            name_dict: Dictionary mapping client IDs to names
            number: Number of clients to select
            fedgra_mode: Selection mode ('used' or 'all_participate')
            min_participate_round: Minimum staleness/rounds for fairness selection
            break_client_set: Set of broken clients to exclude
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of selected client IDs
        """
        if break_client_set is None:
            break_client_set = set()
        else:
            break_client_set = set(break_client_set)
            
        if fedgra_mode == 'used':
            return self._vanilla_selection(grg_matrix, name_dict, number)
        elif fedgra_mode == 'all_participate':
            return self._all_participate_selection(
                grg_matrix, fairness_matrix, name_dict, number, 
                min_participate_round, break_client_set
            )
        else:
            return self._all_participate_selection(
                grg_matrix, fairness_matrix, name_dict, number,
                min_participate_round, break_client_set
            )
    
    def _vanilla_selection(self, grg_matrix, name_dict, number):
        """Vanilla FedGRA selection method."""
        # Determine number of clients to pick
        if number < 1:
            clients_to_pick = max(1, round(number * len(grg_matrix)))
        else:
            clients_to_pick = int(number)
        
        # Convert dict to dataframe and rank by GRG
        # print(type(grg_matrix))
        if not grg_matrix:
            return []
            
        df = pd.DataFrame.from_dict(grg_matrix, orient='index', columns=['Value'])
        client_rank = df.sort_values(ascending=False, by='Value')
        
        # Get selected clients
        selected_client = client_rank[:clients_to_pick].index.tolist()
        # keys = [value for key, value in name_dict.items() if key in selected_client]
        # print(keys)
        
        self._print_selected_clients(selected_client, name_dict)
        return selected_client
    
    def _all_participate_selection(self, grg_matrix, fairness_matrix, name_dict, 
                                 number, min_participate_round, break_client_set):
        """Fairness + GRG: fairness clients always selected, GRG adds extras.

        Fairness clients (staleness >= threshold) are unconditionally selected.
        Additionally, *number* GRG top-k clients are picked from the remaining
        pool.  Total may exceed *number* when fairness clients exist.
        """
        if number < 1:
            clients_to_pick = max(1, round(number * len(grg_matrix)))
        else:
            clients_to_pick = int(number)

        # ── Fairness: staleness >= threshold → always selected ────────
        fairness_clients = [
            key for key, value in fairness_matrix.items()
            if value >= min_participate_round and key not in break_client_set
        ]

        # ── GRG: always pick top-k from the remaining pool ────────────
        fairness_set = set(fairness_clients)
        clients_left = {
            key: value for key, value in grg_matrix.items()
            if key not in fairness_set and key not in break_client_set
        }
        additional = []
        if clients_to_pick > 0 and clients_left:
            df = pd.DataFrame.from_dict(clients_left, orient='index', columns=['Value'])
            client_rank = df.sort_values(ascending=False, by='Value')
            additional = client_rank[:clients_to_pick].index.tolist()

        selected_client = fairness_clients + additional

        self._print_selected_clients(selected_client, name_dict)
        return selected_client

    # -------------------------------
    # Grey Relational Analysis helpers
    # -------------------------------
    def _GRA(self, data, reference_series=None, normalize='max', rho=0.5, weights=None):
        """Grey Relational Analysis (GRA) with metric weights."""

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if normalize == 'max':
            data_normalized = data / data.max()
        elif normalize == 'min-max':
            data_normalized = (data - data.min()) / (data.max() - data.min())
        elif normalize == 'mean':
            data_normalized = data / data.mean()
        else:
            raise ValueError("Parameter 'normalize' should be 'max', 'min-max', or 'mean'.")

        if reference_series is None:
            reference_series = data_normalized.max()
        else:
            if normalize == 'max':
                reference_series = reference_series / data.max()
            elif normalize == 'min-max':
                reference_series = (reference_series - data.min()) / (data.max() - data.min())
            elif normalize == 'mean':
                reference_series = reference_series / data.mean()

        diff_matrix = abs(data_normalized - reference_series)
        delta_min = diff_matrix.min().min()
        delta_max = diff_matrix.max().max()

        relation_coefficient = (delta_min + rho * delta_max) / (diff_matrix + rho * delta_max)

        if weights is None:
            weights = np.ones(data.shape[1]) / data.shape[1]
        else:
            weights = np.array(weights)
            if len(weights) != data.shape[1]:
                raise ValueError("Length of weights must match the number of indicators.")
            weights = weights / weights.sum()

        weights_series = pd.Series(weights, index=data.columns)
        weighted_coefficients = relation_coefficient.mul(weights_series, axis=1)
        relational_grade = weighted_coefficients.sum(axis=1)

        print("Grey Relational Grade:", relational_grade)

        return relational_grade

    def _compute_gra_scores(self, current_cids: List[str]) -> Optional[Dict[str, float]]:
        """Ported directly from KerasFL fedgra.py main loop.

        KerasFL equivalent:
            client_weight_divergence_normalized = MetricCalculator.normalize_positive(client_weight_divergence)
            client_loss_square_normalized       = MetricCalculator.normalize_negative(client_loss_square)
            gra_input = {
                'weight_divergence': client_weight_divergence_normalized,
                'loss':              client_loss_square_normalized
            }
            weights_ewm = MetricCalculator.entropy_weight_method(gra_input)
            gra_grade   = MultiCriteriaDecisionAnalysis.grey_relational_analysis(gra_input, weights=weights_ewm)
            participants_id = gra_grade.sort_values(ascending=False).head(2).index.tolist()
        """
        # ── 1. Collect raw metrics ──────────────────────────────────────────
        ordered_cids = []
        loss_raw = []
        wd_raw   = []
        for cid in current_cids:
            data = self._clients_data_dict.get(cid, {}) or {}
            if not isinstance(data, dict):
                continue
            if self.fedgra_metric_profile == "standard":
                loss = data.get("initial_loss")
                wd = data.get("weight_cosine_distance")
                if not loss:  # None or 0.0 (standard trainer no longer computes this)
                    loss = data.get("keras_avg_loss")
                if wd is None:
                    wd = data.get("weight_l2_delta_keras")
            else:
                # KerasFL metric profile: ported original metrics.
                loss = data.get("initial_loss")
                wd = data.get("weight_l2_delta_keras")
            if not loss or wd is None:
                continue
            ordered_cids.append(cid)
            loss_raw.append(float(loss))
            wd_raw.append(float(wd))

        if len(ordered_cids) < 2:
            return None

        # ── 2. Normalise ───────────────────────────────────────────────────
        # Min-max normalisation:
        #   positive: (x - min) / (max - min)  → higher raw → higher score
        #   negative: (max - x) / (max - min)  → higher raw → lower score
        def normalize_positive(p_list):
            min_p, max_p = min(p_list), max(p_list)
            if max_p == min_p:
                return [1.0] * len(p_list)
            return [(p - min_p) / (max_p - min_p) for p in p_list]

        def normalize_negative(n_list):
            min_n, max_n = min(n_list), max(n_list)
            if max_n == min_n:
                return [1.0] * len(n_list)
            return [(max_n - n) / (max_n - min_n) for n in n_list]

        # Square WD to amplify divergence differences before normalisation
        wd_squared = [w ** 2 for w in wd_raw]

        wd_norm   = normalize_positive(wd_squared)
        loss_norm = normalize_negative(loss_raw)

        # ── 3. EWM — exact KerasFL entropy_weight_method ───────────────────
        # Input: already-normalised lists (KerasFL passes them directly)
        gra_input = {
            'weight_divergence': wd_norm,
            'loss':              loss_norm,
        }

        data_matrix = np.array(list(gra_input.values())).T   # (N_clients, 2)
        num_samples = data_matrix.shape[0]
        eps = np.finfo(float).eps

        norm_data = (data_matrix - data_matrix.min(axis=0)) / (
            data_matrix.max(axis=0) - data_matrix.min(axis=0) + eps
        )
        P       = norm_data / (norm_data.sum(axis=0) + eps)
        k       = 1.0 / np.log(num_samples)
        entropy = -k * np.sum(P * np.log(P + eps), axis=0)
        redundancy  = 1.0 - entropy
        weights_ewm = (redundancy / redundancy.sum()).tolist()

        self._last_ewm_weights = dict(zip(gra_input.keys(), weights_ewm))

        # ── 4. GRA — exact KerasFL grey_relational_analysis ─────────────────
        data_df = pd.DataFrame(gra_input, index=ordered_cids)

        # Step 1: max-value normalisation (KerasFL default normalize='max')
        data_normalized = data_df / data_df.max()

        # Step 2: reference sequence = mean of each column (average baseline)
        reference_series = data_normalized.mean()

        # Step 3: grey relational coefficient
        diff_matrix = abs(data_normalized - reference_series)
        delta_min = diff_matrix.min().min()
        delta_max = diff_matrix.max().max()
        rho = self.fedgra_eps
        relation_coefficient = (delta_min + rho * delta_max) / (diff_matrix + rho * delta_max)

        # Step 4: weighted sum
        w = np.array(weights_ewm)
        w = w / w.sum()
        weights_series = pd.Series(w, index=data_df.columns)
        relational_grade = relation_coefficient.mul(weights_series, axis=1).sum(axis=1)

        print("FedGRA Weights:", self._last_ewm_weights)
        print("Grey Relational Grade:", relational_grade)

        return relational_grade.to_dict()

    def _normalize_p(self, values: List[float]) -> List[float]:
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [1.0 for _ in values]
        return [(v - v_min) / (v_max - v_min) for v in values]

    def _normalize_n(self, values: List[float]) -> List[float]:
        v_min = min(values)
        v_max = max(values)
        if v_max == v_min:
            return [1.0 for _ in values]
        return [(v_max - v) / (v_max - v_min) for v in values]

    def _EWM(self, metric_matrix, drop_max_min=True):
        """
        Entropy Weight Method (EWM) — matches the original Keras implementation.

        Parameters
        ----------
        metric_matrix : dict
            ``{client_id: {metric_name: value, ...}, ...}``
        drop_max_min : bool
            Unused — kept for signature compatibility. The original Keras
            implementation does not drop max/min rows.

        Returns
        -------
        pd.Series
            Normalised weights indexed by metric name, summing to 1.

        Algorithm (identical to original Keras ``entropy_weight_method``):
        1. Build data matrix  (rows = clients, cols = metrics)
        2. Normalise each column to [0, 1]
        3. Compute proportion  P_ij = x_ij / Σ_i x_ij
        4. Compute entropy     E_j  = -(1/ln N) * Σ_i P_ij * ln(P_ij)
        5. Redundancy          d_j  = 1 - E_j
        6. Weight              w_j  = d_j / Σ_j d_j
        """
        # Build {metric: [values across clients]} preserving client order
        ordered_cids = list(metric_matrix.keys())
        metric_names = list(next(iter(metric_matrix.values())).keys())

        data_dict = {
            m: [metric_matrix[cid][m] for cid in ordered_cids]
            for m in metric_names
        }

        # Rows = clients, cols = metrics  (matches original: data.T)
        data = np.array(list(data_dict.values()), dtype=float).T
        num_samples = data.shape[0]

        eps = np.finfo(float).eps

        # 1. Normalise to [0, 1]
        col_min = data.min(axis=0)
        col_max = data.max(axis=0)
        norm_data = (data - col_min) / (col_max - col_min + eps)

        # 2. Proportion
        P = norm_data / (norm_data.sum(axis=0) + eps)

        # 3. Entropy
        k = 1.0 / np.log(num_samples) if num_samples > 1 else 1.0
        entropy = -k * np.sum(P * np.log(P + eps), axis=0)

        # 4. Redundancy & weights
        redundancy = 1.0 - entropy
        if redundancy.sum() == 0:
            weights = np.ones(len(metric_names)) / len(metric_names)
        else:
            weights = redundancy / redundancy.sum()

        return pd.Series(weights, index=metric_names, name="ewm_weight")

    def _print_selected_clients(self, selected: List[str], name_dict: Dict[str, str], prefix: str = "FedGRA"):
        """Helper to print selected clients"""
        pass
        # names = [name_dict.get(cid, cid) for cid in selected]
        # print(f"{prefix}: Selected {len(selected)} - {names}")
