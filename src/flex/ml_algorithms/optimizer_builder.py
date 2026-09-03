from __future__ import annotations
import re
import torch
import torch.optim as optim

from ..ml_utils import Handlers


def _parse_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "t"}: return True
        if s in {"false", "0", "no", "n", "f"}: return False
    raise ValueError(f"Invalid boolean value: {v!r}")

def _parse_betas(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (float(v[0]), float(v[1]))
    if isinstance(v, str):
        s = re.sub(r"[()\[\]\s]", "", v)  # remove (),[], spaces
        parts = s.split(",")
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    raise ValueError(f"Invalid betas value: {v!r}. Expect (0.9,0.999)")

def _filter_params(params):
    """支持两种形式：iter(parameters) 或 param_groups(list[dict])."""
    # param_groups
    if isinstance(params, (list, tuple)) and all(isinstance(g, dict) for g in params):
        new_groups = []
        for g in params:
            ps = g.get("params", [])
            ps = [p for p in ps if getattr(p, "requires_grad", True)]
            if len(ps) == 0:  # 跳过空组
                continue
            ng = {k: v for k, v in g.items() if k != "params"}
            ng["params"] = ps
            new_groups.append(ng)
        return new_groups if new_groups else [{"params": []}]
    # plain iterable
    return [p for p in params if getattr(p, "requires_grad", True)]


class OptimizerBuilder(Handlers):
    """
    Auto-build a PyTorch optimizer from config.

    config_dict 示例:
      {"optimizer": {
          "type": "adamw",
          "lr": 1e-3,
          "weight_decay": 1e-4,
          "betas": "(0.9,0.999)",
          "eps": 1e-8,
          "amsgrad": "false"
      }}
    """

    def __init__(self, parameters, config_dict):
        super().__init__()
        self.config = config_dict.get("optimizer", config_dict)
        self.parameters = parameters
        self._optimizer: optim.Optimizer | None = None

        # 注册
        self.register_handler("sgd", self.__build_sgd)
        self.register_handler("adam", self.__build_adam)
        self.register_handler("adamw", self.__build_adamw)   # 新增
        self.register_handler("adagrad", self.__build_adagrad)
        self.register_handler("rmsprop", self.__build_rmsprop)

    def build(self, filter_requires_grad: bool = True) -> optim.Optimizer:
        opt_type = str(self.config.get("type", "sgd")).lower()

        if "lr" not in self.config:
            raise ValueError("Learning rate 'lr' must be specified in optimizer config.")
        kwargs = {"lr": float(self.config.get("lr"))}

        # 过滤冻结参数
        params = _filter_params(self.parameters) if filter_requires_grad else self.parameters

        optimizer = self.invoke_handler(opt_type, kwargs=kwargs, params=params)
        if optimizer is None:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        self._optimizer = optimizer
        return optimizer

    def rebuild(self, new_parameters, filter_requires_grad: bool = True) -> optim.Optimizer:
        self.parameters = new_parameters
        return self.build(filter_requires_grad=filter_requires_grad)

    # ---------- private helpers ----------
    def __safe_add(self, kwargs: dict, key: str, caster=None):
        if key not in self.config: 
            return
        value = self.config[key]
        if value is None or (isinstance(value, str) and value.strip().lower() == "none"):
            return
        if caster is not None:
            value = caster(value)
        kwargs[key] = value

    # ---------- builders ----------
    def __build_sgd(self, kwargs, params) -> optim.Optimizer:
        self.__safe_add(kwargs, "momentum", float)
        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "nesterov", _parse_bool)
        # 兜底：nesterov=True 必须 momentum>0
        if kwargs.get("nesterov", False) and kwargs.get("momentum", 0.0) <= 0.0:
            raise ValueError("SGD with nesterov=True requires momentum > 0.")
        return optim.SGD(params, **kwargs)

    def __build_adam(self, kwargs, params) -> optim.Optimizer:
        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "eps", float)
        self.__safe_add(kwargs, "amsgrad", _parse_bool)
        self.__safe_add(kwargs, "betas", _parse_betas)
        self._maybe_enable_fused(kwargs, params)
        return optim.Adam(params, **kwargs)

    def __build_adamw(self, kwargs, params) -> optim.Optimizer:
        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "eps", float)
        self.__safe_add(kwargs, "amsgrad", _parse_bool)
        self.__safe_add(kwargs, "betas", _parse_betas)
        self._maybe_enable_fused(kwargs, params)
        return optim.AdamW(params, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _maybe_enable_fused(kwargs: dict, params=None) -> None:
        """Enable fused Adam/AdamW only when safe (CUDA params)."""
        from ..ml_utils.training_utils import TrainingUtils
        if not TrainingUtils.is_optimization_enabled("fused_optimizer"):
            return
        if not torch.cuda.is_available():
            return
        # Only enable fused if ALL params are on CUDA
        if params is not None:
            try:
                all_cuda = True
                for p in params:
                    if not p.is_cuda:
                        all_cuda = False
                        break
                if all_cuda:
                    kwargs["fused"] = True
            except Exception:
                pass

    def __build_adagrad(self, kwargs, params) -> optim.Optimizer:
        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "eps", float)
        return optim.Adagrad(params, **kwargs)

    def __build_rmsprop(self, kwargs, params) -> optim.Optimizer:
        self.__safe_add(kwargs, "momentum", float)
        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "alpha", float)
        self.__safe_add(kwargs, "centered", _parse_bool)
        self.__safe_add(kwargs, "eps", float)
        return optim.RMSprop(params, **kwargs)


# from __future__ import annotations

# import torch.optim as optim

# from ..ml_utils import Handlers


# class OptimizerBuilder(Handlers):
#     """
#     A class that automatically constructs a PyTorch optimizer based on the configuration dictionary.

#     Args:
#         parameters: The model parameters (typically use model.parameters()).
#         config_dict: A configuration dictionary (choose either config_path or config_dict).
#     """

#     def __init__(self, parameters, config_dict):
#         super().__init__()

#         if "optimizer" in config_dict:
#             self.config = config_dict["optimizer"]
#         else:
#             self.config = config_dict

#         self.parameters = parameters
#         self._optimizer: optim.Optimizer

#         # Register standard method
#         self.register_handler("sgd", self.__build_sgd)
#         self.register_handler("adam", self.__build_adam)
#         self.register_handler("adagrad", self.__build_adagrad)
#         self.register_handler("rmsprop", self.__build_rmsprop)                
#         return


#     def build(self) -> optim.Optimizer:
#         """
#         build optimizer return optim.Optimizer
#         """

#         optimizer_type = self.config.get("type", "sgd").lower()

#         #Check lr in dict
#         if "lr" not in self.config:
#             raise ValueError("Learning rate 'lr' must be specified in optimizer config.")

#         kwargs = {"lr": float(self.config.get("lr"))}
#         optimizer = self.invoke_handler(optimizer_type, kwargs=kwargs)
#         if optimizer is None:
#             raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

#         return optimizer


#     def rebuild(self, new_parameters) -> optim.Optimizer:
#         """
#         rebuild optimizer when needed
#         """

#         self.parameters = new_parameters
#         self._optimizer = self.build()
#         return self._optimizer

#     ##################################################################
#     # private functions

#     def __safe_add(self, kwargs: dict, key: str, cast_type = None):
#         value = self.config.get(key)
#         if value is None or str(value).lower() == "none":
#             return

#         if cast_type is not None:
#             try:
#                 value = cast_type(value)
#             except Exception as e:
#                 raise ValueError(f"Failed to cast optimizer config '{key}' to {cast_type}: {e}")

#         kwargs[key] = value
#         return


#     def __build_sgd(self, kwargs) -> optim.Optimizer:
#         self.__safe_add(kwargs, "momentum", float)
#         self.__safe_add(kwargs, "nesterov", bool)
#         self.__safe_add(kwargs, "weight_decay", float)
#         return optim.SGD(self.parameters, **kwargs)

#     def __build_adam(self, kwargs) -> optim.Optimizer:
#         self.__safe_add(kwargs, "weight_decay", float)
#         self.__safe_add(kwargs, "eps", float)
#         self.__safe_add(kwargs, "amsgrad", bool)
#         self.__safe_add(kwargs, "betas", lambda v: tuple(map(float, v)))
#         return optim.Adam(self.parameters, **kwargs)

#     def __build_adagrad(self, kwargs) -> optim.Optimizer:
#         self.__safe_add(kwargs, "weight_decay", float)
#         self.__safe_add(kwargs, "eps", float)
#         return optim.Adagrad(self.parameters, **kwargs)

#     def __build_rmsprop(self, kwargs) -> optim.Optimizer:
#         self.__safe_add(kwargs, "momentum", float)
#         self.__safe_add(kwargs, "weight_decay", float)
#         self.__safe_add(kwargs, "alpha", float)
#         self.__safe_add(kwargs, "centered", bool)
#         self.__safe_add(kwargs, "eps", float)
#         return optim.RMSprop(self.parameters, **kwargs)
