import json, os
from typing import Any

import torch
import torch.nn as nn
import yaml

from .lora import LoRALinear
from ..ml_utils import console, Handlers


class ModelExtractor(Handlers):
    """
    Extract Model params
    """

    def __init__(self):
        super().__init__()

        self.last_extracted_data: dict = {}

        # Register default LoRA extractor
        self.register_handler(LoRALinear, self.__extract_handler_lora_linear)
        return

    def register_handler(self, any_key: Any, handler_fn):
        """
        Register any type's handler function callback
        Model extract handler function format:
            def <func_name>(<model>: nn.Module, <only_trainable>: bool = False) -> dict
        """

        if self.exists_handler(any_key):
            console.warn(f"Layer type({any_key}) already exists, will replace previous handler.")

        super().register_handler(any_key, handler_fn)
        return self

    def extract_layers(self, model: nn.Module, only_trainable = False) -> dict:
        """
        Extract model all layers data dictionary

        Args:
            model: NN model
        return:
            layer data(dictionary) extracted
        """

        layer_data_dict = {}

        for name, module in model.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue

            extracted = None

            for custom_type, handler_fn in self.handlers.items():
                if isinstance(module, custom_type) and handler_fn is not None:
                    extracted = handler_fn(module)
                    break

            if extracted is None:
                extracted = {
                    k: v.detach().cpu()
                    for k, v in module.named_parameters(recurse=False)
                    if not only_trainable or v.requires_grad
                }

                # 添加默认类型字段
                if extracted:
                    extracted["layer_type"] = type(module).__name__.lower()

            if extracted:
                layer_data_dict[name] = extracted

        self.last_extracted_data = layer_data_dict
        return layer_data_dict

    # Export lay data to file
    def export_to_file(self, file_path: str):
        """
        Export model layer data to file
        """
        if file_path.endswith("json"):
            self.__export_to_json(file_path)
        else:
            self.__export_to_npz_or_yaml(file_path)

        return

    #private
    def __export_to_json(self, filepath: str):
        serializable = {
            layer: {
                name: tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in params.items()
            }
            for layer, params in self.last_extracted_data.items()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent = 2)

        return

    #private
    def __export_to_npz_or_yaml(self, filepath: str):
        flat_data = {}

        for layer, params in self.last_extracted_data.items():
            for name, tensor in params.items():
                if isinstance(tensor, torch.Tensor):
                    key = f"{layer}.{name}"
                    flat_data[key] = tensor.numpy()

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if filepath.endswith(".npz"):
            np_save_path = filepath if filepath.endswith(".npz") else filepath + ".npz"
            torch.save(flat_data, np_save_path)
        else:
            with open(filepath, "w") as f:
                yaml.dump(flat_data, f, indent = 2)
        return

    #------------------------------------
    # Default extract handlers
    #------------------------------------
    # private
    def __extract_handler_lora_linear(self, module, only_trainable: bool = False):
        return {
            "weight": module.weight.detach().cpu(),
            "bias": module.bias.detach().cpu() if module.bias is not None else None,
            "lora_A": module.lora_A.detach().cpu(),
            "lora_B": module.lora_B.detach().cpu(),
            "layer_type": "lora",
        }