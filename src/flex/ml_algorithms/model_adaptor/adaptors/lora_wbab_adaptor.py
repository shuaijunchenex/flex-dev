from typing import Any
import torch.nn as nn
from model_adaptor.model_adaptor_abc import WeightAdapter

class ModelAdaptor_LoraWbab(WeightAdapter):
    def __init__(self, source_weights=None):
        super().__init__()
        self.source_weights = source_weights

    def set_source_weights(self, weights_dict: dict) -> 'ModelAdaptor_LoraWbab':
        self.source_weights = weights_dict
        return self

    def load_from_extractor(self, extractor: Any) -> 'ModelAdaptor_LoraWbab':
        self.source_weights = extractor.get_layer_dict()
        return self

    def load_from_file(self, filepath: str) -> 'ModelAdaptor_LoraWbab':
        import torch, json
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                self.source_weights = json.load(f)
        elif filepath.endswith('.npz'):
            loaded = torch.load(filepath)
            restructured = {}
            for key, value in loaded.items():
                parts = key.split('.')
                layer_name = '.'.join(parts[:-1])
                param_name = parts[-1]
                if layer_name not in restructured:
                    restructured[layer_name] = {}
                restructured[layer_name][param_name] = torch.tensor(value)
            self.source_weights = restructured
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")
        return self

    def apply_to_model(self, target_model: nn.Module) -> nn.Module:
        if self.source_weights is None:
            raise ValueError("未设置源权重，请先使用set_source_weights()、load_from_extractor()或load_from_file()设置")
        return self.apply_weights_to_model(target_model, self.source_weights)

    @staticmethod
    def assign_or_expand(target_tensor, source_tensor, layer_type):
        import numpy as np, torch
        source = source_tensor.detach().cpu().numpy() if isinstance(source_tensor, torch.Tensor) else np.array(source_tensor)
        target_shape = target_tensor.shape
        source_shape = source.shape

        if layer_type in ['lora', 'batchnorm2d', 'conv2d'] and len(source_shape) == len(target_shape) + 1:
            if source_shape[0] == target_shape[0] or source_shape[0] == 1:
                source = source.reshape(source_shape[1:])
            else:
                raise ValueError(f"源形状多出一维，但与目标首维不匹配: {source_shape} vs {target_shape}")

        if len(source.shape) != len(target_shape):
            raise ValueError(f"源/目标张量维度不一致: {source.shape} vs {target_shape}")

        result = np.zeros(target_shape, dtype=source.dtype)
        if len(target_shape) == 4:
            slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target_shape))
            result[slices] = source[slices]
        elif len(target_shape) == 2:
            min_rows, min_cols = min(source.shape[0], target_shape[0]), min(source.shape[1], target_shape[1])
            result[:min_rows, :min_cols] = source[:min_rows, :min_cols]
        elif len(target_shape) == 1:
            min_len = min(source.shape[0], target_shape[0])
            result[:min_len] = source[:min_len]
        else:
            raise NotImplementedError(f"不支持的张量维度: {target_shape}")

        tensor_to_copy = torch.tensor(result, dtype=target_tensor.dtype, device=target_tensor.device)
        target_tensor.copy_(tensor_to_copy)
        return target_tensor

    @staticmethod
    def apply_weights_to_model(model, extracted_weights):
        import torch.nn as nn
        from lora.impl.lora_linear import LoRALinear
        model_layers = {
            name: module
            for name, module in model.named_modules()
            if len(list(module.parameters(recurse=False))) > 0
        }

        for layer_name, layer_data in extracted_weights.items():
            if layer_name not in model_layers:
                print(f"[警告] 目标模型中找不到层: {layer_name}")
                continue

            target_module = model_layers[layer_name]
            layer_type = layer_data.get("layer_type", "").lower()

            def copy_param(attr_name):
                if hasattr(target_module, attr_name) and attr_name in layer_data:
                    target_attr = getattr(target_module, attr_name)
                    ModelAdaptor_LoraWbab.assign_or_expand(target_attr.data, layer_data[attr_name], layer_type)

            if layer_type == "lora":
                copy_param("weight")
                copy_param("bias")
                copy_param("lora_A")
                copy_param("lora_B")
            elif layer_type == "batchnorm2d":
                copy_param("weight")
                copy_param("bias")
                copy_param("running_mean")
                copy_param("running_var")
            elif layer_type == "conv2d":
                copy_param("weight")
                copy_param("bias")
            else:
                for param_name, param_value in layer_data.items():
                    if param_name == "layer_type":
                        continue
                    if hasattr(target_module, param_name):
                        target_attr = getattr(target_module, param_name)
                        if isinstance(target_attr, nn.Parameter) or isinstance(target_attr, torch.Tensor):
                            ModelAdaptor_LoraWbab.assign_or_expand(target_attr.data, param_value, layer_type)
                    else:
                        full_name = f"{layer_name}.{param_name}"
                        if full_name in model.state_dict():
                            param_tensor = model.state_dict()[full_name]
                            ModelAdaptor_LoraWbab.assign_or_expand(param_tensor.data, param_value, layer_type)
                        else:
                            print(f"[提示] 未在模型中找到参数: {full_name}")
        return model
