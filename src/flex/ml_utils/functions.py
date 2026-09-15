from __future__ import annotations


def dict_get(dictionary: dict, similar_keys: str, default=None):
    """
    Dictionary get string key value, support similar means.

    Arg:
        dict: string key dictionary
        similar_keys: combined similar key string
            key format: "<key-1>|<key-2>|<key-3>...", eg.
        default: when key not exist, return default.
    Return: key value or default

    Sample:
           value = dict_get(config_dict, "data_loader|dataset_loader")
       i.e. get key 'data_loader' or 'dataset_loader' value from config_dict
    """
    for s in similar_keys.split("|"):
        s = s.strip()
        if s in dictionary:
            return dictionary[s]
    return default


def dict_exists(dictionary: dict, similar_keys: str) -> bool:
    """
    Dictionary exists string key, support similar means.

    Arg:
        dict: string key dictionary
        similar_keys: combined similar key string
            key format: "<key-1>|<key-2>|<key-3>...", e.g.
    """
    for s in similar_keys.split("|"):
        s = s.strip()
        if s in dictionary:
            return True

    return False
