from __future__ import annotations
from typing import Any

from ..string import String
from .event_args import EventArgs


class EventHandler:
    """
    Event Handler(delegate) class. 
    - Event must declare before attach callback function
    - Multi handler support
    """

    def __init__(self):
        """
        Event function handler(callback) dict
        """
        self.__event_handler_dict: dict[str, Any] = {}
        self.__event_declare_list: list = []     # declare event names
        return

    def declare_events(self, *event_names: str):
        """
        Declare event names
        """
        for name in event_names:
            if String.is_none_or_empty(name) or name in self.__event_declare_list:
                continue
            self.__event_declare_list.append(name)
        return self
    
    def is_event_declared(self, event_name: str) -> bool:
        """
        Check if event_name declared
        """
        if String.is_none_or_empty(event_name): 
            raise ValueError("Event name can not be empty or None.")

        return event_name in self.__event_declare_list

    def exists_event(self, event_name: str, handler=None) -> bool:
        """
        Event exists
        """
        if String.is_none_or_empty(event_name): 
            raise ValueError("Event name can not be empty or None.")

        # Check if event is declared
        if not self.is_event_declared(event_name):
            return False

        # Check event attached
        if event_name not in self.__event_handler_dict:
            return False

        if handler is not None:
            return handler in self.__event_handler_dict[event_name]
        return True

    def attach_event(self, event_name: str, handler):
        """
        Attach event handler to event name
        """
        if String.is_none_or_empty(event_name): 
            raise ValueError("Event name can not be empty or None.")

        # Check if event is declared
        if not self.is_event_declared(event_name):
            raise ValueError(f"Event name '{event_name}' must declare first.")
        
        if handler is None:
            raise ValueError(f"Attach event handler must not None.")
        
        if not callable(handler):
            raise ValueError(f"Attach event handler must be callable.")

        if not self.exists_event(event_name):
            self.__event_handler_dict[event_name] = [handler]
            return self

        if self.exists_event(event_name, handler):
            return self

        self.__event_handler_dict[event_name].append(handler)
        return self

    def detach_event(self, event_name: str, handler=None):
        """
        Detach event handler
        """
        if String.is_none_or_empty(event_name): 
            raise ValueError("Event name can not be empty or None.")
        
        if not self.exists_event(event_name, handler):
            return self

        if handler is None:
            del self.__event_handler_dict[event_name]  # Remove all handler
        else:
            self.__event_handler_dict[event_name].remove(handler)
            if len(self.__event_handler_dict[event_name]) <= 0:
                del self.__event_handler_dict[event_name]
        return self

    def raise_event(self, event_name: str, args: EventArgs|None):
        """
        Raise event
        """
        if String.is_none_or_empty(event_name): 
            raise ValueError("Event name can not be empty or None.")
        
        # Check if event is declared
        if not self.is_event_declared(event_name):
            raise ValueError(f"Event name '{event_name}' must declare first.")

        if not self.exists_event(event_name):
            return

        for handler in self.__event_handler_dict[event_name]:
            if handler is not None:
                handler(args)
        return

    def clear_event(self):
        """
        " Clear all handler
        """
        self.__event_handler_dict.clear()

    def __str__(self):
        return " ".join(self.__event_handler_dict)
