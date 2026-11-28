from __future__ import annotations
from typing import Any


class ObjectMap(object):
    """
    Object dictionary
    """

    def __init__(self):
        self.__object_map: dict = {}    # private
        return

    @property
    def count(self):
        """
        Number of object added
        """
        return len(self.__object_map)

    def remove_object(self, any_key: Any):
        """
        Remove object by key
        """
        if any_key is None:
            return
        
        if any_key in self.__object_map:
            del self.__object_map[any_key]
        return self

    def exists_object(self, any_key):
        """
        Determine fn key is registered
        """
        if any_key is None:
            raise ValueError("Object key is None.")
        return any_key in self.__object_map

    def set_object(self, any_key: Any, object_instance: Any):
        """
        Set or replace object instance
        """
        if any_key is None:
            raise ValueError("Object key is None.")
        self.__object_map[any_key] = object_instance
        return

    def get_object(self, any_key: Any, default_value=None, cast_type=None) -> Any:
        """
        get object by key, with safe 'cast_type':
        - If cast_type is a class/type: 
            * return obj if isinstance(obj, cast_type)
            * try conversion only for simple casters (int/float/str/bool/dict/list/tuple/set)
            * otherwise raise TypeError (avoid calling base classes like nn.Module(obj))
        - If cast_type is a callable (converter function), call it with obj.
        """
        import inspect
        from typing import Any

        if any_key is None:
            raise ValueError("Object key is None")
        if not self.exists_object(any_key):
            return default_value

        obj = self.__object_map[any_key]

        if cast_type is None:
            return obj

        try:
            # 1) cast_type 是“类型/类”
            if inspect.isclass(cast_type):
                # 已经是目标类型，直接返回
                if isinstance(obj, cast_type):
                    return obj

                if inspect.isclass(obj) and issubclass(obj, cast_type):
                    try:
                        return obj()  # 仅当无参构造可行
                    except Exception as e:
                        raise TypeError(
                            f"Stored class '{obj.__name__}' is a subclass of {cast_type.__name__} "
                            f"but failed to instantiate without args: {e}"
                        ) from e

                # 其他复杂类型（如 nn.Module）不做“强转”，直接报类型不匹配
                raise TypeError(
                    f"Object under key '{any_key}' is {type(obj).__name__}, "
                    f"expected {cast_type.__name__} (isinstance check failed; no safe cast)."
                )

            # 2) cast_type 是“转换器函数”
            if callable(cast_type):
                return cast_type(obj)

            # 3) 其他情况：非法的 cast_type
            raise TypeError(f"cast_type must be a class or callable, got {type(cast_type).__name__}")

        except Exception as e:
            # 报错信息里带上对象与目标类型，便于定位
            raise ValueError(
                f"Failed to get key '{any_key}': cannot conform object {obj!r} "
                f"({type(obj).__name__}) to {cast_type}: {e}"
            ) from e

    # def get_object(self, any_key: Any, default_value=None, cast_type=None) -> Any:
    #     """
    #     get object by key
    #     """
    #     if any_key is None:
    #         raise ValueError(f"Object key is None")
    #     if not self.exists_object(any_key):
    #         return default_value

    #     if cast_type is not None:
    #         try:
    #             return cast_type(self.__object_map[any_key]) #TODO: check cast type
    #         except Exception as e:
    #             raise ValueError(f"Failed to cast instance '{self.__object_map[any_key]}' to {cast_type}: {e}")

    #     return self.__object_map[any_key]
