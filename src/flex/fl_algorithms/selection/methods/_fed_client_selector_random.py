from __future__ import annotations
import random

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_Random(FedClientSelector):
    """
    Random client selection, optionally using a private seeded RNG.

    ``independent_rng=True`` isolates participation from other code's calls to
    Python's global RNG. Nonnegative seeds (including zero) are used directly;
    a negative seed requests a nondeterministic stream. The default retains
    the historical global RNG and seed conversion for existing experiments.
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        self._args.select_method = "random"
        return

    def with_random_seed(self, seed: int = -1):
        # The base constructor invokes this after assigning self._args. Avoid
        # its global random.seed call entirely for the opt-in private stream.
        if self._args.independent_rng:
            self._rng = random.Random(None if seed < 0 else seed)
            return self
        return super().with_random_seed(seed)

    def select(self, client_list: list, select_number: int = -1):
        """
        Select clients from client list,
        Args:
            select_number: if selection_number <= 0, select number from args.select_number,
                           else use selection_number
        """        
        if select_number <= 0:
            select_number = self._args.select_number

        # If available clients are less than or equal to requested number, select all
        if len(client_list) <= select_number:
            return client_list

        sampler = self._rng if self._args.independent_rng else random
        return sampler.sample(client_list, select_number)
