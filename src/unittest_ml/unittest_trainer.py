from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms

# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.model_trainer import ModelTrainerFactory
from flex.ml_utils import ConfigLoader, console
from flex.ml_algorithms import LoRALinear, LossFunctionBuilder, OptimizerBuilder
from flex.ml_data_loader import DatasetLoaderFactory

# 定义 LoRAMLP 模型
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

# 使用 ModelTrainer 训练 LoRAMLP 模型
def train_with_trainer(yaml):

    # trainer_args = ModelTrainerArgs(yaml)
    trainer_args = ModelTrainerFactory.create_args(yaml)

    data_loader_args = DatasetLoaderFactory.create_args(yaml)
    data_loader_args.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(torch.flatten)
    ])

    # Override some params manually if necessary
    data_loader_args.root = "../../../.dataset"
    data_loader = DatasetLoaderFactory.create(data_loader_args)

    # add some params to args
    trainer_args.train_loader = data_loader.data_loader
    trainer_args.device = "cpu"

    epochs = 5
    rank = 60

    # 初始化模型
    model = LoRAMLP(input_dim=28*28, hidden_dim=256, output_dim=10, rank=rank).to(trainer_args.device)

    # 损失函数 & 优化器
    criterion = LossFunctionBuilder.build(yaml)
    optimizer = OptimizerBuilder(model.parameters(), yaml).build()

    # 使用 ModelTrainer 训练
    trainer_args.model = model
    trainer_args.optimizer = optimizer
    trainer_args.loss_func = criterion

    trainer = ModelTrainerFactory.create(trainer_args)
    state_dict, stats = trainer.train(epochs)

    return state_dict, stats
    

def main():

    # trainer args in yaml
    yaml_file_name = './test_data/test_trainer.yaml'
    yaml = ConfigLoader.load(yaml_file_name)

    console.out("------------- Begin ---------------")
    result = train_with_trainer(yaml)
    console.info(result)
    console.out("------------- End -----------------")
    return

if __name__ == "__main__":
    main()
