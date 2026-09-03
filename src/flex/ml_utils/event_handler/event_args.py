from __future__ import annotations
from typing import Any


class EventArgs:
    """
    Event Args class
    """

    def __init__(self, sender: Any = None, kind: str = "", data: Any = None):
        """
        Event sender object
        Members:
            sender: who call the event
            data: event data
            kind: event kind string
        """
        self._sender: Any = sender
        self.kind: str = kind
        self.data: Any = data

    @property
    def sender(self) -> Any:
        """
        " Readonly sender object
        """
        return self._sender

    def with_kind(self, kind: str) -> EventArgs:
        """
        " set event kind
        """
        self.kind = kind
        return self

    def with_data(self, data: Any) -> EventArgs:
        """
        " set event kind
        """
        self.data = data
        return self

    def with_sender(self, sender: Any) -> EventArgs:
        """
        " set event arg with sender object
        """
        self._sender = sender
        return self
