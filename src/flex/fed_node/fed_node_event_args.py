from __future__ import annotations

from ..ml_utils import EventArgs

'''
' Fed node args event args
'''

class FedNodeEventArgs(EventArgs):
    def __init__(self, kind: str, config_dict: dict):
        super().__init__()

        """
        Operation kind
        """
        self.kind: str = kind

        """
        Config dict
        """
        self.config_dict: dict = config_dict

    @property
    def client_args(self):
        return self.sender

    @property
    def server_args(self):
        return self.sender

