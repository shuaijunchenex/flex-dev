from __future__ import annotations
from abc import ABC, abstractmethod

import random
from typing import Any
import torch

from .fed_aggregator_args import FedAggregatorArgs


class AbstractFedAggregator(ABC):
    """
    Fed Server Aggregator class interface declare
    """
    
    def __init__(self, args: FedAggregatorArgs|None = None):
        """
        Initialize aggregator with updated weight list and aggregation method.

        Arg:
            args(FedAggregatorArgs): aggregator configuration arguments.
        """

        if args is None:
            self.args = FedAggregatorArgs()
        else:
            self.args = args

        # data list
        self._aggregation_data_dict: dict = {}

        # Select method
        self._aggregation_method = ""

        # Aggregated weight
        self._aggregated_weight: Any = None  # can be wbab or torch dict or ....

        # perform aggregation on ('cpu' or 'cuda')
        self._device = torch.device(self.args.device)
        return

    @property       # aggregated weight property
    def aggregated_weight(self): return self._aggregated_weight

    @property       # aggregation method property
    def aggregated_method(self): return self._aggregation_method

    def with_random_seed(self, seed: int):
        """
        Manual set random seed, Return self
        """
        random.seed(seed)
        return self

    def with_clients_update(self, clients_update: Any):
        """
        When select clients according to some client data

        Args:
            client_data: any clients data, maybe dict or list

        Return:
            self
        """
        self._clients_update = clients_update
        return self

    def aggregate(self, client_data_dict):
        """
        Aggregate the client updates, Return aggregated weight
        """
        self._aggregation_data_dict: list = [(d["updated_weights"], d["train_record"]["data_sample_num"]) for d in client_data_dict]  # [[model_weight: dict / wbab, vol],[model_weight: dict / wbab, vol]]
        self._before_aggregation()
        self._do_aggregation()
        self._after_aggregation()
        return self._aggregated_weight

    ########################################
    # Abstract method

    @abstractmethod
    def _before_aggregation(self) -> None:
        """
        Call before aggregate
        """
        pass

    @abstractmethod
    def _do_aggregation(self) -> None:
        """
        do aggregate
        """
        pass

    @abstractmethod
    def _after_aggregation(self) -> None:
        """
        Call after aggregate
        """
        pass
