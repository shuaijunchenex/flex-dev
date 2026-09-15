import torch
import torch.nn as nn
import json

from lora.impl.lora_linear import LoRALinear

class LoRAModelWeightAdapter:
    """
    适配不同大小和结构的PyTorch模型权重的工具类。
    可以用于将一个模型的权重转移到另一个具有类似但不完全相同架构的模型上。
    """
    
    def __init__(self, source_weights=None):
        """
        初始化模型权重适配器
        
        Args:
            source_weights (dict, optional): 预先提取的权重字典，可选
        """
        self.source_weights = source_weights
    
    @staticmethod
    def assign_or_expand(target_tensor, source_tensor, layer_type):
        """
        将源张量数据适配并复制到目标张量，支持不同层的形状对齐。
        会直接修改 target_tensor 的值（in-place），不会创建新张量。

        Args:
            target_tensor (torch.Tensor): 目标张量，将被覆盖（in-place）
            source_tensor (torch.Tensor or array-like): 源张量或数组
            layer_type (str): 层类型（如 lora、conv2d、batchnorm2d）

        Returns:
            torch.Tensor: 被修改过的 target_tensor（同一个对象）
        """
        import numpy as np
        # 将 source 转换为 numpy
        source = source_tensor.detach().cpu().numpy() if isinstance(source_tensor, torch.Tensor) else np.array(source_tensor)
        target_shape = target_tensor.shape
        source_shape = source.shape

        # 特殊处理：source 多出一维时（如 LoRA 扩展）
        if layer_type in ['lora', 'batchnorm2d', 'conv2d'] and len(source_shape) == len(target_shape) + 1:
            if source_shape[0] == target_shape[0] or source_shape[0] == 1:
                source = source.reshape(source_shape[1:])
            else:
                raise ValueError(f"源形状多出一维，但与目标首维不匹配: {source_shape} vs {target_shape}")

        # 再次检查形状兼容性
        if len(source.shape) != len(target_shape):
            raise ValueError(f"源/目标张量维度不一致: {source.shape} vs {target_shape}")

        # 创建一个中间 numpy 结果（零填充）
        result = np.zeros(target_shape, dtype=source.dtype)
        if len(target_shape) == 4:
            # Conv2D 权重
            slices = tuple(slice(0, min(s, t)) for s, t in zip(source.shape, target_shape))
            result[slices] = source[slices]
        elif len(target_shape) == 2:
            # Linear 权重
            min_rows, min_cols = min(source.shape[0], target_shape[0]), min(source.shape[1], target_shape[1])
            result[:min_rows, :min_cols] = source[:min_rows, :min_cols]
        elif len(target_shape) == 1:
            # 偏置/BN 参数
            min_len = min(source.shape[0], target_shape[0])
            result[:min_len] = source[:min_len]
        else:
            raise NotImplementedError(f"不支持的张量维度: {target_shape}")

        # in-place 拷贝数据到 target_tensor
        tensor_to_copy = torch.tensor(result, dtype=target_tensor.dtype, device=target_tensor.device)
        target_tensor.copy_(tensor_to_copy)
        return target_tensor  # 保持接口一致

    @staticmethod
    def apply_weights_to_model(model, extracted_weights):
        """
        将提取的权重应用到目标模型（in-place 修改）。

        Args:
            model (nn.Module): 要加载权重的目标 PyTorch 模型
            extracted_weights (dict): 从另一个模型提取的权重字典

        Returns:
            nn.Module: 应用权重后的原模型（同一个对象）
        """
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
                    LoRAModelWeightAdapter.assign_or_expand(target_attr.data, layer_data[attr_name], layer_type)

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
                # 通用处理
                for param_name, param_value in layer_data.items():
                    if param_name == "layer_type":
                        continue

                    if hasattr(target_module, param_name):
                        target_attr = getattr(target_module, param_name)
                        if isinstance(target_attr, nn.Parameter) or isinstance(target_attr, torch.Tensor):
                            LoRAModelWeightAdapter.assign_or_expand(target_attr.data, param_value, layer_type)
                    else:
                        # 查找 state_dict 中的名称作为最后的 fallback
                        full_name = f"{layer_name}.{param_name}"
                        if full_name in model.state_dict():
                            param_tensor = model.state_dict()[full_name]
                            LoRAModelWeightAdapter.assign_or_expand(param_tensor.data, param_value, layer_type)
                        else:
                            print(f"[提示] 未在模型中找到参数: {full_name}")

        return model
    
    def set_source_weights(self, weights_dict):
        """
        设置源权重字典
        
        Args:
            weights_dict (dict): 源模型权重字典
            
        Returns:
            PyTorchModelWeightAdapter: 返回自身以支持链式调用
        """
        self.source_weights = weights_dict
        return self
    
    def load_from_extractor(self, extractor):
        """
        从AdvancedModelExtractor加载权重
        
        Args:
            extractor (AdvancedModelExtractor): 模型提取器实例
            
        Returns:
            PyTorchModelWeightAdapter: 返回自身以支持链式调用
        """
        self.source_weights = extractor.get_layer_dict()
        return self
    
    def load_from_file(self, filepath):
        """
        从文件加载权重数据
        
        Args:
            filepath (str): 文件路径，支持.json和.npz格式
            
        Returns:
            PyTorchModelWeightAdapter: 返回自身以支持链式调用
        """
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                self.source_weights = json.load(f)
        elif filepath.endswith('.npz'):
            loaded = torch.load(filepath)
            # 重构为层级结构
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
    
    def apply_to_model(self, target_model):
        """
        将源权重应用到目标模型
        
        Args:
            target_model (nn.Module): 要应用权重的PyTorch模型
            
        Returns:
            nn.Module: 更新后的模型
        """
        if self.source_weights is None:
            raise ValueError("未设置源权重，请先使用set_source_weights()、load_from_extractor()或load_from_file()设置")
        
        return self.apply_weights_to_model(target_model, self.source_weights)
    
    @staticmethod
    def set_wbab(model: nn.Module, wbab: dict):
        """
        Load weights from structured wbab dict into a given model.

        Args:
            wbab (dict): Layer-wise structured weight dict.
            model (nn.Module): The target PyTorch model to update.
        """
        for name, module in model.named_modules():
            if name in wbab:
                params = wbab[name]
                for param_name, value in params.items():
                    if param_name == "layer_type":
                        continue  # skip type tag

                    try:
                        # safer way to construct tensor
                        if isinstance(value, torch.Tensor):
                            tensor_value = value.clone().detach()
                        else:
                            tensor_value = torch.tensor(value)

                        # === LoRA-specific ===
                        if isinstance(module, LoRALinear):
                            if param_name == "weight":
                                module.weight.data.copy_(tensor_value)
                            elif param_name == "lora_A":
                                module.lora_A.data.copy_(tensor_value)
                            elif param_name == "lora_B":
                                module.lora_B.data.copy_(tensor_value)
                            elif param_name == "bias" and module.bias is not None:
                                module.bias.data.copy_(tensor_value)

                        # === BatchNorm-specific ===
                        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                            if hasattr(module, param_name):
                                setattr(module, param_name, tensor_value)

                        # === Default layer ===
                        elif hasattr(module, param_name):
                            param = getattr(module, param_name)
                            if isinstance(param, torch.nn.Parameter):
                                param.data.copy_(tensor_value)
                            else:
                                setattr(module, param_name, tensor_value)

                    except Exception as e:
                        print(f" Failed to load {name}.{param_name}: {e}")

