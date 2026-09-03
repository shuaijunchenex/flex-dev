from __future__ import annotations
from typing import Any


class MultiHandlers:
    """
    Multi handlers
    """
    def __init__(self):
        self.__handler_dict = {}
        return

    def exists_handler(self, any_key: Any, handler=None) -> bool:
        """
        " handler exists
        """
        if any_key not in self.__handler_dict:
            return False

        if handler is not None:
            return handler in self.__handler_dict[any_key]
        return True

    def register_handler(self, any_key: Any, handler):
        """
        Register handler to a key
        """
        if not self.exists_handler(any_key):
            self.__handler_dict[any_key] = [handler]
            return self

        if self.exists_handler(any_key, handler):
            return self

        self.__handler_dict[any_key].append(handler)
        return self

    def unregister_handler(self, any_key: Any, handler = None):
        """
        Unregister a key handler
        """
        if not self.exists_handler(any_key, handler):
            return self

        if handler is None:
            del self.__handler_dict[any_key]  # Remove key's all handler
        else:
            self.__handler_dict[any_key].remove(handler)
            if len(self.__handler_dict[any_key]) <= 0:
                del self.__handler_dict[any_key]
        return self

    def invoke_handler(self, any_key: Any, *args, **kwargs):
        """
        Invoke handler
        """
        if not self.exists_handler(any_key):
            return

        for handler in self.__handler_dict[any_key]:
            if handler is not None:
                handler(*args, **kwargs)

    def clear_handlers(self):
        """
        " Clear all handler
        """
        self.__handler_dict.clear()

    def __str__(self):
        return " ".join(self.__handler_dict)
