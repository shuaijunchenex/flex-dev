import torch

from collections import OrderedDict
from ..fed_aggregator_abc import AbstractFedAggregator, FedAggregatorArgs

from ....ml_utils import console


class FedAggregator_FedAvg(AbstractFedAggregator):
    """
    Implements the FedAvg aggregation method using weighted average of client updates.
    """

    def __init__(self, args: FedAggregatorArgs|None = None):
        super().__init__(args)
        self._aggregation_method = "fedavg"
        return

    def build_data_list(self, aggregation_data_dict: dict) -> None:
        """
        Build the aggregation data list from the provided dictionary.
        Each entry in the dictionary should be a tuple of (model_weights, data_volume).
        """
        self._aggregation_data_list = list(aggregation_data_dict.values())
        return

    # override
    def _before_aggregation(self) -> None:
        #self.build_data_list(self._aggregation_data_dict)
        return
        
    # override
    def _do_aggregation(self) -> None:
        console.debug(f"\n[FedAvg] Starting aggregation with {len(self._aggregation_data_dict)} clients...")
        sample_state_dict = self._aggregation_data_dict[0][0]  # get one sample model
        new_weights = OrderedDict()

        # Init empty tensor for aggregation.
        # Use float32 for integer-dtype params (e.g. num_batches_tracked) so that
        # weighted accumulation does not trigger a Float->Long cast error.
        for key in sample_state_dict.keys():
            ref = sample_state_dict[key]
            if ref.is_floating_point():
                new_weights[key] = torch.zeros_like(ref, device=self._device)
            else:
                new_weights[key] = torch.zeros(ref.shape, dtype=torch.float32, device=self._device)

        total_data_vol = sum(vol for _, vol in self._aggregation_data_dict)

        # Debug info
        for i, (_, vol) in enumerate(self._aggregation_data_dict):
            console.debug(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        for state_dict, vol in self._aggregation_data_dict:
            weight = vol / total_data_vol
            for key in new_weights:
                param = state_dict[key].to(self._device)
                if param.is_floating_point():
                    # Floating-point params (weights, biases, running_mean/var): weighted average
                    new_weights[key] += param * weight
                else:
                    # Integer params (e.g. BatchNorm num_batches_tracked): accumulate weighted sum
                    # then round back to integer at the end via _after_aggregation.
                    # Use float accumulation buffer to avoid Long cast error.
                    new_weights[key] += param.float() * weight

        # Restore integer-dtype params (e.g. num_batches_tracked) to their original dtype
        for key in new_weights:
            orig_dtype = sample_state_dict[key].dtype
            if not sample_state_dict[key].is_floating_point():
                new_weights[key] = new_weights[key].round().to(orig_dtype)

        self._aggregated_weight = new_weights

        # Debug: first param mean
        first_param_name = next(iter(new_weights.keys()))
        console.debug(f"[FedAvg] Aggregated first param mean: {new_weights[first_param_name].mean():.6f}")
        return

    # override
    def _after_aggregation(self) -> None:
        console.debug(f"[FedAvg] Aggregation completed.")
