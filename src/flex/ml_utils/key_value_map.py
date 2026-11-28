from .string import String
from .dict_path import DictPath, get_dict_value
from typing import Any


class KeyValueMap:
    """
    String K-V dictionary, use dot(.) as key path
    """
    def __init__(self, from_dict: dict[str, Any], is_clone_dict = False):
        # private
        self._key_value_dict = DictPath(from_dict, is_clone_dict)

        # key name to dictionary key path map
        self._keys_map = KeyValueMap.build_key_map(from_dict, "/")   # All dict items
        return

    @property
    def key_value_dict(self): return self._key_value_dict

    # private
    def __map_get(self, key, default_value):
        return self._key_value_dict.get(self._keys_map[key], default_value)

    def get(self, key: str, default_value: Any = None):
        """
        Get value from similar key name, can use path like 'key/key1'
        """
        for k in key.split("|"):
            k = k.strip()
            if k in self._keys_map:
                return self.__map_get(k, default_value)

        return default_value

    def set(self, key: str, value: Any):
        """
        Set value from key name
        """
        self._key_value_dict.set(self._keys_map[key], value)
        return

    def exists(self, key: str):
        """
        Key name existence
        """
        return key in self._keys_map

    #--------------------------------------------------
    # Default key_name split char
    __CONST_SPLIT_CHAR = '.'

    @staticmethod
    def set_split_char(ch: str):
        """
        Set split char of key name
        """
        KeyValueMap.__CONST_SPLIT_CHAR = ch

    @staticmethod
    def build_key_map(from_dict: dict[str, Any], start_key_path = "/"):

        if start_key_path == "/" or String.is_none_or_empty(start_key_path):
            start_dict = from_dict
            start_key_path = ""
        else:
            start_dict = get_dict_value(from_dict, start_key_path)

        m = KeyValueMap.__build_one_key_map(start_dict, start_key_path, "")
        for k in m:
            m[k] = k.replace(KeyValueMap.__CONST_SPLIT_CHAR, "/")

        return m

    @staticmethod
    def __build_one_key_map(start_dict, start_key_path, up_key_name):

        # Inner function
        def __get_key_name(kn, name):
            if String.is_none_or_empty(kn) or kn == "/":
                return name
            else:
                return f"{kn}{KeyValueMap.__CONST_SPLIT_CHAR}{name}"

        m = {}
        if isinstance(start_dict, list):
            lst: list = list(start_dict)
            for index in range(len(lst)):
                key_name = __get_key_name(up_key_name, index)
                v = KeyValueMap.__build_one_key_map(lst[index], f"{start_key_path}/{index}", key_name)
                m.update(v)
        else:
            if not isinstance(start_dict, dict):
                return m

            for k, _ in start_dict.items():
                key_name = __get_key_name(up_key_name, k)
                if isinstance(start_dict[k], dict):
                    v = KeyValueMap.__build_one_key_map(start_dict[k], k, key_name)
                    m.update(v)
                elif isinstance(start_dict[k], list):
                    v = KeyValueMap.__build_one_key_map(start_dict[k], k, key_name)
                    m.update(v)
                else:
                    m[key_name] = f"{start_key_path}/{k}"

        return m
