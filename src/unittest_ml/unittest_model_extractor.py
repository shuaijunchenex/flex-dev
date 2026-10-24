from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import console, ConfigLoader
from flex.ml_algorithms import ModelExtractor, LoRALinear, LoRAArgs


class LoRAMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank=4):
        super().__init__()
        self.fc1 = LoRALinear(input_dim, hidden_dim, rank=rank)
        self.relu = nn.ReLU()
        self.fc2 = LoRALinear(hidden_dim, output_dim, rank=rank)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 示例模型（CNN + LoRALinear）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = LoRALinear(32 * 5 * 5, 10, rank=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# 示例模型（CNN + LoRALinear）
class SimpleCNN_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = LoRALinear(32 * 5 * 5, 10, rank=5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

def test_model_extractor():

    # Form yaml file
    yaml_file_name = './test_data/node_config_template_client.yaml'
    console.out(f"Test from yaml file: {yaml_file_name}")
    yaml = ConfigLoader.load(yaml_file_name)

    # 假设你已经有一个包含 LoRALinear 层的模型
    lora_args = LoRAArgs(yaml)                  # 注意：args可用可不用

    lora_args.set("input_dim", 10).set("hidden_dim", 16).set("output_dim", 5).set("rank", 2)

    model = LoRAMLP(input_dim=10, hidden_dim=16, output_dim=5, rank=2)

    # 提取器初始化
    extractor = ModelExtractor()

    #Extract all layer
    layer_params = extractor.extract_layers(model, False)
    console.info(layer_params)

    extractor.export_to_file("./.results/layer_params.json")
    extractor.export_to_file("./.results/layer_params.npz")
    return

def test_model_extractor_1():
    # 创建模型和提取器
    model = SimpleCNN()
    extractor = ModelExtractor()

# ✅ 注册 BatchNorm 提取器
    def extract_batchnorm_fn(module: nn.modules.batchnorm._BatchNorm):
        return {
            "weight": module.weight.detach().cpu(),
            "bias": module.bias.detach().cpu(),
            "running_mean": module.running_mean.detach().cpu(),
            "running_var": module.running_var.detach().cpu(),
            "num_batches_tracked": module.num_batches_tracked.item(),
            "layer_type": "batchnorm"
        }

    extractor.register_handler(nn.BatchNorm2d, extract_batchnorm_fn)
    layer_dict = extractor.extract_layers(model)

    # 输出结构化字典
    for layer_name, params in layer_dict.items():
        print(f"\nLayer: {layer_name}")
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {v}")

    layer_dict_2 = extractor.extract_layers(SimpleCNN_2())

    console.ok("Extraction finished.")
    return


def main():
    test_model_extractor()
    test_model_extractor_1()
    return

if __name__ == "__main__":
    main()
