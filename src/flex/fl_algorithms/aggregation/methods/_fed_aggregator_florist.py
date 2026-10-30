"""FLoRIST aggregation from MLSys 2026, Algorithm 1.

Inputs use A[r, in] and B[out, r]. The strategy must absorb each client's
LoRA scaling into B before upload; these factors represent effective updates.
Only explicitly trainable head parameters may accompany the adapters.

Reference: https://proceedings.mlsys.org/paper_files/paper/2026/file/
f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf
This implementation follows the equations independently of the upstream
training code (DASS-Lab-Group/FLoRIST, revision
100d373025866d148e092a17001e1232dabeeabd).
"""

from collections import OrderedDict
from collections.abc import Mapping
import math
from numbers import Real
import time

import torch

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs


class FedAggregator_Florist(AbstractFedAggregator):
    def __init__(self, args: FedAggregatorArgs | None = None):
        super().__init__(args)
        threshold = self.args.get("threshold", 0.9)
        if (isinstance(threshold, bool) or not isinstance(threshold, Real)
                or not math.isfinite(threshold) or not 0 < threshold <= 1):
            raise ValueError("FLoRIST threshold must be finite and in (0, 1].")
        self.threshold = float(threshold)
        self._aggregation_method = "florist"
        self.last_stats = {}

    def aggregate(self, client_data_dict):
        self.last_stats = {}
        self._aggregated_weight = None
        if not isinstance(client_data_dict, (list, tuple)) or not client_data_dict:
            raise ValueError("FLoRIST requires a nonempty list of client updates.")
        for update in client_data_dict:
            if (not isinstance(update, Mapping) or "updated_weights" not in update
                    or not isinstance(update.get("train_record"), Mapping)
                    or "data_sample_num" not in update["train_record"]):
                raise ValueError("Each client requires updated_weights and train_record.data_sample_num.")
        return super().aggregate(client_data_dict)

    def _before_aggregation(self):
        pairs = self._aggregation_data_dict
        states = [state for state, _ in pairs]
        counts = [count for _, count in pairs]
        if any(isinstance(n, bool) or not isinstance(n, Real)
               or not math.isfinite(n) or n < 0 for n in counts):
            raise ValueError("Client sample counts must be finite nonnegative numbers.")
        try:
            total = math.fsum(counts)
        except OverflowError as exc:
            raise ValueError("Total client sample count must be finite.") from exc
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Total client sample count must be positive and finite.")
        self._sample_weights = [float(n) / total for n in counts]
        if any(not isinstance(state, Mapping) or not state for state in states):
            raise ValueError("Client updated_weights must be nonempty tensor mappings.")
        keys = list(states[0])
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("Parameter keys must be strings.")
        if any(set(state) != set(keys) for state in states):
            raise ValueError("All clients must provide the same parameter keys.")
        for key in keys:
            if "lora_" in key and not key.endswith((".lora_A", ".lora_B")):
                raise ValueError(f"Unsupported LoRA key: {key}")
            for state in states:
                value = state[key]
                if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
                        or value.layout != torch.strided):
                    raise ValueError(f"Expected a dense real floating tensor for {key}.")
                # if not torch.isfinite(value).all().item():
                #     raise ValueError(f"Nonfinite tensor: {key}")
            if key.endswith((".lora_A", ".lora_B")):
                other = key[:-1] + ("B" if key.endswith("A") else "A")
                if other not in states[0]:
                    raise ValueError(f"Missing LoRA partner for {key}.")
                if any(state[key].ndim != 2 or min(state[key].shape) <= 0 for state in states):
                    raise ValueError(f"LoRA factors must be nonempty matrices: {key}")
            elif any(state[key].shape != states[0][key].shape for state in states):
                raise ValueError(f"Head parameter shapes disagree: {key}")
        self._prefixes = [key[:-7] for key in keys if key.endswith(".lora_A")]
        if not self._prefixes:
            raise ValueError("FLoRIST requires at least one LoRA factor pair.")
        for prefix in self._prefixes:
            a_key, b_key = prefix + ".lora_A", prefix + ".lora_B"
            in_dim, out_dim = states[0][a_key].shape[1], states[0][b_key].shape[0]
            for state in states:
                a, b = state[a_key], state[b_key]
                if a.shape[0] != b.shape[1]:
                    raise ValueError(f"LoRA A/B ranks disagree: {prefix}")
                if a.shape[1] != in_dim or b.shape[0] != out_dim:
                    raise ValueError(f"LoRA input/output dimensions disagree: {prefix}")

    @staticmethod
    def _dtype(values):
        # CPU SVD does not support fp16/bfloat16. Preserve double-precision
        # references; ordinary training inputs use fp32 aggregation.
        return torch.float64 if any(v.dtype == torch.float64 for v in values) else torch.float32

    @staticmethod
    def _nbytes(values):
        return sum(value.numel() * value.element_size() for value in values)

    def _sync(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @torch.no_grad()
    def _do_aggregation(self):
        self._sync()
        started = time.perf_counter()
        states = [state for state, _ in self._aggregation_data_dict]
        weights = self._sample_weights
        result, layers = OrderedDict(), {}
        for key in states[0]:
            if key.endswith((".lora_A", ".lora_B")):
                continue
            values = [state[key] for state in states]
            dtype = self._dtype(values)
            averaged = torch.zeros(values[0].shape, dtype=dtype, device=self._device)
            for value, weight in zip(values, weights):
                averaged.add_(value.detach().to(device=self._device, dtype=dtype), alpha=weight)
            # if not torch.isfinite(averaged).all().item():
            #     raise ValueError(f"Nonfinite averaged head parameter: {key}")
            result[key] = averaged

        for prefix in self._prefixes:
            self._sync()
            layer_started = time.perf_counter()
            a_key, b_key = prefix + ".lora_A", prefix + ".lora_B"
            values = [state[key] for state in states for key in (a_key, b_key)]
            dtype = self._dtype(values)
            a_stack = torch.cat([
                state[a_key].detach().to(device=self._device, dtype=dtype) * weight
                for state, weight in zip(states, weights)
            ], dim=0)
            b_stack = torch.cat([
                state[b_key].detach().to(device=self._device, dtype=dtype) for state in states
            ], dim=1)
            # Two thin factorizations followed by the small core SVD. Never
            # construct B_stack @ A_stack (the dense global update).
            u_b, s_b, vh_b = torch.linalg.svd(b_stack, full_matrices=False)
            u_a, s_a, vh_a = torch.linalg.svd(a_stack, full_matrices=False)
            core = (s_b[:, None] * (vh_b @ u_a)) * s_a[None, :]
            # if not torch.isfinite(core).all().item():
            #     raise ValueError(f"Nonfinite FLoRIST core: {prefix}")
            u, singular, vh = torch.linalg.svd(core, full_matrices=False)
            if singular[0].item() == 0:
                rank, retained, error = 1, 1.0, 0.0
                b_global = torch.zeros((b_stack.shape[0], 1), dtype=dtype, device=self._device)
                a_global = torch.zeros((1, a_stack.shape[1]), dtype=dtype, device=self._device)
            else:
                # Scaling before squaring avoids energy overflow/underflow.
                energy = (singular / singular[0]).square()
                total = energy.sum()
                if self.threshold == 1.0:
                    rank = singular.numel()
                else:
                    cumulative = energy.cumsum(0) / total
                    rank = min(int(torch.searchsorted(cumulative, self.threshold).item()) + 1,
                               singular.numel())
                retained = float((energy[:rank].sum() / total).item())
                error = math.sqrt(float((energy[rank:].sum() / total).item()))
                b_global = (u_b @ u[:, :rank]) * singular[:rank]
                a_global = vh[:rank] @ vh_a
            # if not (torch.isfinite(a_global).all().item() and torch.isfinite(b_global).all().item()):
            #     raise ValueError(f"Nonfinite global LoRA factors: {prefix}")
            result[a_key], result[b_key] = a_global, b_global
            self._sync()
            layers[prefix] = {
                "input_stack_rank": a_stack.shape[0], "global_rank": rank,
                "retained_energy": retained, "relative_truncation_error": error,
                "upload_bytes": self._nbytes(values),
                "download_bytes": self._nbytes((a_global, b_global)),
                "aggregation_seconds": time.perf_counter() - layer_started,
            }
        self._sync()
        self.last_stats = {
            "threshold": self.threshold, "num_clients": len(states), "layers": layers,
            "upload_bytes": sum(self._nbytes(state.values()) for state in states),
            "download_bytes": self._nbytes(result.values()),
            "aggregation_seconds": time.perf_counter() - started,
        }
        self._aggregated_weight = result

    def _after_aggregation(self):
        pass
