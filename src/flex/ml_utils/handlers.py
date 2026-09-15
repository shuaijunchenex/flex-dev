from __future__ import annotations
from inspect import isfunction, ismethod
from typing import Any


class Handlers(object):
    def __init__(self):
        self.__handlers: dict = {}
        return

    @property
    def count(self):
        """
        Number of fn registered
        """
        return len(self.__handlers)

    @property
    def handlers(self):
        return self.__handlers

    def register_handler(self, any_key: Any, handler_fn):
        """
        Register handler function callback, if handler function exists, replace the old one
        """
        if any_key is None:
            raise ValueError("Handler key is None.")

        if handler_fn is None:
            raise ValueError("Register handler is None.")

        if not (isfunction(handler_fn) or ismethod(handler_fn) or callable(handler_fn)):
            raise ValueError("Register handler is not callable.")

        self.__handlers[any_key] = handler_fn
        return self

    def unregister_handler(self, any_key: Any):
        """
        Unregister handler
        """
        if any_key is None:
            raise ValueError("Handler key is None.")
        
        if any_key in self.__handlers:
            del self.__handlers[any_key]
        return self

    def exists_handler(self, any_key):
        """
        Determine fn key is registered
        """
        if any_key is None:
            raise ValueError("Handler key is None.")
        
        return any_key in self.__handlers

    def invoke_handler(self, any_key: Any, *args, **kwargs):
        """
        Execute fn call
        """
        if any_key is None:
            raise ValueError("Handler key is None.")
        
        if not self.exists_handler(any_key):
            return None

        fn = self.__handlers[any_key]
        if fn is not None:
            return fn(*args, **kwargs)

    def safe_add_kwarg(self, kwargs: dict, key: str, value: Any, cast_type = None):
        """
        Safely add an arg to kwargs
        """
        if cast_type is not None:
            try:
                value = cast_type(value)
            except Exception as e:
                raise ValueError(f"Failed to cast optimizer config '{key}' to {cast_type}: {e}")

        kwargs[key] = value
        return
