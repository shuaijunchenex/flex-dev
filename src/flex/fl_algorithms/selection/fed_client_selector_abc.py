from __future__ import annotations
from abc import ABC, abstractmethod

import datetime
import random
from typing import Any

from .fed_client_selector_args import FedClientSelectorArgs


class FedClientSelector(ABC):
    def __init__(self, args: FedClientSelectorArgs|None = None):        
        """
        Initialize Selector with args
        """
        if args is None:
            self._args = FedClientSelectorArgs()
        else:
            self._args = args

        self._clients_data_dict: dict = {}          # Client data dictionary
        if args is not None:
            self.with_random_seed(args.random_seed)
        return

    @property
    def select_method(self): return self._args.select_method

    @property
    def select_number(self): return self._args.select_number

    @property
    def client_selection_round(self): return self._args.client_selection_round

    @property
    def select_round(self): return self._args.select_round

    def with_random_seed(self, seed: int = -1):
        """
        Manual set random seed, -1 means random seed generate by time
        """
        if seed <= 0:
            dt = datetime.datetime.now()
            seed = dt.microsecond % 1000        # range: 0~999
        random.seed(int(seed * 1000) % 1000)
        return self

    def with_clients_data(self, clients_data_dict: dict):
        """
        When select clients according to some client data
        """
        self._clients_data_dict = clients_data_dict
        return self

    @abstractmethod
    def select(self, client_list: list, select_number: int = -1) -> list:
        """
        Select clients from client list,
        if selection_number <= 0, select number from args.select_number, else use selection_number
        """
        return []
