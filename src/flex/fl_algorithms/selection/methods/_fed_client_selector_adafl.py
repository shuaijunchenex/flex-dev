from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_AdaFL(FedClientSelector):
    """
    Adaptive federated learning client selector (AdaFL).
    Keeps a decayed cumulative contribution score per client.
    """

    def __init__(self, args: FedClientSelectorArgs | None = None):
        super().__init__(args)
        self._args.select_method = "adafl"
        self._cfg = self._load_cfg()
        self._round_index = 0
        # Instance-level state (must NOT be class variables to avoid cross-experiment contamination)
        self._cumulative_v: Dict[Any, float] = {}
        self._avg_loss: float = 1.0
        self._loss_count: int = 0

    def _load_cfg(self) -> Dict[str, float]:
        """Load AdaFL hyper-parameters from args if provided."""
        adafl_block = self._args.get("adafl", None)

        def _get_raw(key: str, default: Any) -> Any:
            # Priority 1: nested block (client_selection.adafl.<key>)
            # Priority 2: flat key (client_selection.<key>) for backward compatibility
            raw_nested = None
            if adafl_block is not None:
                try:
                    raw_nested = adafl_block.get(key, None)
                except Exception:
                    raw_nested = None

            if raw_nested is not None:
                return raw_nested

            raw_flat = self._args.get(key, None)
            if raw_flat is not None:
                return raw_flat

            return default

        def _get_int(key: str, default: int, minimum: Optional[int] = None) -> int:
            raw = _get_raw(key, default)
            try:
                val = int(raw)
            except (TypeError, ValueError):
                val = default
            if minimum is not None:
                val = max(minimum, val)
            return val

        def _get_float(key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
            raw = _get_raw(key, default)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = default
            if min_val is not None:
                val = max(min_val, val)
            if max_val is not None:
                val = min(max_val, val)
            return val

        m0 = _get_int("m0", 2, minimum=1)
        T0 = _get_int("T0", 5, minimum=1)
        deltaT = _get_int("deltaT", 1, minimum=1)
        m_max = _get_int("m_max", _get_int("mmax", 20, minimum=1), minimum=1)
        beta = _get_float("beta", 0.7, min_val=1e-6, max_val=0.999999)

        return {
            "m0": m0,
            "T0": T0,
            "deltaT": deltaT,
            "m_max": m_max,
            "beta": beta,
        }

    def _calc_m_t(self, t: int) -> int:
        m0 = self._cfg["m0"]
        T0 = self._cfg["T0"]
        deltaT = self._cfg["deltaT"]
        m_max = self._cfg["m_max"]

        if t <= T0:
            return min(m0, m_max)

        inc = math.floor((t - T0) / deltaT + m0)
        return min(inc, m_max)

    def _theta(self, t: int) -> float:
        beta = self._cfg["beta"]
        theta = 1.0 - (beta ** t)
        return max(0.0, min(1.0, theta))

    def _extract_loss(self, data: Any) -> Optional[float]:
        if not isinstance(data, dict):
            return None

        # Priority 1: Try trainer-returned metrics (avg_loss, train_loss_sum, etc.)
        # These are now at top level after server extracts from nested train_record
        for key in ("avg_loss", "train_loss_sum", "keras_avg_loss"):
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    pass
        
        # Priority 2: Check epoch_loss list and use the last value
        epoch_loss = data.get("epoch_loss")
        if isinstance(epoch_loss, list) and len(epoch_loss) > 0:
            try:
                return float(epoch_loss[-1])  # Use final epoch loss
            except (TypeError, ValueError):
                pass

        # Priority 3: Legacy keys for backward compatibility
        for key in ("loss", "train_loss", "training_loss"):
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    pass

        # Priority 4: Check nested train_record (for backward compatibility)
        record = data.get("train_record") if isinstance(data, dict) else None
        if isinstance(record, dict):
            for key in ("avg_loss", "train_loss_sum", "loss", "sqrt_train_loss_power_two_sum"):
                if key in record and record[key] is not None:
                    try:
                        return float(record[key])
                    except (TypeError, ValueError):
                        pass

        return None

    def _extract_volume(self, cid: str, client_obj: Any, data: Any) -> float:
        n_k: Optional[float] = None

        if isinstance(data, dict):
            for key in ("data_volume", "data_sample_num", "n_k"):
                if key in data and data[key] is not None:
                    try:
                        n_k = float(data[key])
                        break
                    except (TypeError, ValueError):
                        continue

        if n_k is None and client_obj is not None:
            if hasattr(client_obj, "data_sample_num"):
                try:
                    n_k = float(client_obj.data_sample_num)
                except (TypeError, ValueError):
                    n_k = None
            elif hasattr(client_obj, "node_var") and getattr(client_obj, "node_var") is not None:
                node_var = getattr(client_obj, "node_var")
                if hasattr(node_var, "data_sample_num"):
                    try:
                        n_k = float(node_var.data_sample_num)
                    except (TypeError, ValueError):
                        n_k = None

        if n_k is None or n_k <= 0:
            n_k = 1.0

        return n_k

    def _build_client_info(self, client_ids: List[str], client_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        info: List[Dict[str, Any]] = []

        for cid in client_ids:
            # Try both string and raw client id keys for stored metrics
            raw_cid = cid
            data = self._clients_data_dict.get(raw_cid)
            if data is None and raw_cid not in self._clients_data_dict:
                try:
                    alt_key = int(cid)
                    data = self._clients_data_dict.get(alt_key)
                except (TypeError, ValueError):
                    data = None

            loss_val = self._extract_loss(data)
            volume_val = self._extract_volume(cid, client_map.get(cid), data)

            info.append({
                "client_id": cid,
                "data_volume": volume_val,
                "loss": loss_val,
            })

        return info

    def _select_ids(self, client_ids: List[str], client_info: List[Dict[str, Any]], break_client_set: Set[str], t: int, m_t: int) -> List[str]:
        loss_lookup: Dict[str, Optional[float]] = {}
        loss_values: List[float] = []

        for entry in client_info:
            cid = str(entry.get("client_id"))
            loss_val = self._get_loss(entry)
            loss_lookup[cid] = loss_val
            if loss_val is not None:
                loss_values.append(loss_val)

        if loss_values:
            total_loss = sum(loss_values)
            self._avg_loss = (self._avg_loss * self._loss_count + total_loss) / (self._loss_count + len(loss_values))
            self._loss_count += len(loss_values)

        current_v: Dict[str, float] = {}
        for entry in client_info:
            cid = entry.get("client_id")
            cid_str = str(cid) if cid is not None else None
            if cid_str is None or cid_str in break_client_set:
                continue
            n_k = entry.get("data_volume", 1.0)
            loss_val = loss_lookup.get(cid_str)
            if loss_val is None:
                loss_val = self._avg_loss
            v_t = math.sqrt(float(n_k)) * float(loss_val)
            current_v[cid_str] = v_t

        theta = self._theta(t)
        for cid, v_t in current_v.items():
            prev = self._cumulative_v.get(cid, 0.0)
            self._cumulative_v[cid] = (1.0 - theta) * prev + theta * v_t

        for cid in client_ids:
            cid_str = str(cid)
            if cid_str in break_client_set:
                continue
            if cid_str not in self._cumulative_v:
                self._cumulative_v[cid_str] = self._avg_loss

        scores = self._cumulative_v
        ranked = sorted(
            [str(cid) for cid in client_ids if str(cid) not in break_client_set],
            key=lambda c: scores.get(c, 0.0),
            reverse=True,
        )

        limit = min(m_t, len(ranked))
        return ranked[:limit]

    def _get_loss(self, entry: Dict[str, Any]) -> Optional[float]:
        for key in ("loss", "train_loss", "training_loss"):
            if key in entry and entry[key] is not None:
                try:
                    return float(entry[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _print_selected_clients(self, selected: List[str], name_dict: Dict[str, Any], prefix: str = "AdaFL"):
        names = [name_dict.get(cid, cid) for cid in selected]
        try:
            print(f"{prefix}: selected {len(selected)} -> {names}")
        except Exception:
            pass

    def select(self, client_list: list, select_number: int = -1, **kwargs) -> list:
        if select_number <= 0:
            select_number = self._args.select_number

        if not client_list:
            return []

        client_ids = [str(client.node_id) for client in client_list]
        client_map = {str(client.node_id): client for client in client_list}
        name_dict = {str(client.node_id): getattr(client, "name", str(client.node_id)) for client in client_list}
        break_client_set = {str(cid) for cid in kwargs.get("break_client_set", set())}

        round_from_kw = kwargs.get("round_number", kwargs.get("round"))
        if round_from_kw is None:
            self._round_index += 1
            t = self._round_index
        else:
            try:
                t = int(round_from_kw) + 1
            except (TypeError, ValueError):
                self._round_index += 1
                t = self._round_index

        m_t = self._calc_m_t(t)
        # Do NOT cap m_t by select_number here: AdaFL's adaptive size IS the
        # selection mechanism. The yaml "number" field is ignored for AdaFL;
        # use m_max inside the adafl config block to cap the ceiling instead.
        m_t = min(m_t, len(client_ids))
        if m_t <= 0:
            return []

        client_info = self._build_client_info(client_ids, client_map)
        selected_ids = self._select_ids(client_ids, client_info, break_client_set, t, m_t)

        self._print_selected_clients(selected_ids, name_dict)
        try:
            theta = self._theta(t)
            print(f"AdaFL: t={t}, m_t={m_t}, theta={theta:.4f}")
        except Exception:
            pass

        return [client_map[cid] for cid in selected_ids if cid in client_map]
