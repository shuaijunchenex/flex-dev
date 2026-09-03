from .dict_path import DictPath
from typing import Any


class KeyValueArgs:
    """
    String K-V dictionary args
    """
    def __init__(self, from_dict: dict[str, Any]|None=None, is_clone_dict=False):
        # private
        self.__key_value_dict: DictPath

        if from_dict is None:
            from_dict = {}
        self.set_args(from_dict, is_clone_dict)
        return

    # Getter
    @property
    def key_value_dict(self): return self.__key_value_dict

    def set_args(self, from_dict: dict[str, Any], is_clone_dict=False):
        self.__key_value_dict = DictPath(from_dict, is_clone_dict)

    def get(self, key_name: str, default_value: Any = None) -> Any:
        """
        Get value from similar key name, can use path like 'key/key1'
        """
        for s in key_name.split("|"):
            s = s.strip()
            if s in self.__key_value_dict:
                return self.__key_value_dict.get(s, default_value)

        return default_value

    def set(self, key_name: str, value: Any):
        """
        Set value from key name
        """
        if self.__key_value_dict is dict:
            self.__key_value_dict[key_name] = value
        else:
            self.__key_value_dict.set(key_name, value)
        return self

    def exists(self, key_name: str):
        """
        Key name existence
        """
        return key_name in self.__key_value_dict
