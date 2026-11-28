from __future__ import annotations
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, default_collate


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import ConfigLoader, console
from flex.ml_algorithms import OptimizerBuilder
from flex.fed_strategy.server_strategy_impl._lora_svd_analysis_server import LoraSvdAnalysisServerStrategy

##############################################
# Define a simple model

model = torch.nn.Linear(10, 2)

def test_from_file():
    # Load config from yaml file
    yaml_file = "./test_data/node_config_template_client.yaml"
    config = ConfigLoader.load(yaml_file)
    console.info(f"Yaml file: {yaml_file}")

    optimizer = OptimizerBuilder(model.parameters(), config["optimizer"]).build()
    console.info(f"Optimizer created from configuration file:\n{optimizer}")
    return

def test_direct():

    # Method 2: Provide a configuration dictionary directly
    config = {
        "optimizer": {
            "type": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "momentum": None,
            "nesterov": None,
            "betas": [0.9, 0.999],
            "amsgrad": False,
            "eps": 1e-8,
            "alpha": None,
            "centered": None
        }
    }

    optimizer = OptimizerBuilder(model.parameters(), config_dict = config["optimizer"]).build()
    console.info(f"Optimizer created from configuration dictionary:\n{optimizer}")
    return


@pytest.mark.parametrize("config", [{"type": "sgd", "lr": 0.02, "momentum": 0.8},
                                    {"type": "adam", "lr": 0.02}])
def test_svd_probe_leaves_next_training_step_unchanged(config):
    def collate(batch):
        x, y = default_collate(batch)
        return x + random.random() + np.random.random(), y

    def train(run_probe):
        torch.manual_seed(12)
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.4), nn.Linear(4, 2))
        params = list(model.parameters())
        builder = OptimizerBuilder(params, config)
        optimizer = builder.build()
        loss = nn.CrossEntropyLoss()
        data = TensorDataset(torch.randn(8, 4), torch.arange(8) % 2)
        loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(16))
        nv = SimpleNamespace(model=model, model_weight=model.state_dict(), device="cpu", loss_func=loss,
                             optimizer_builder=builder, data_loader=SimpleNamespace(data_loader=loader))
        probe = LoraSvdAnalysisServerStrategy(None, SimpleNamespace(node_var=nv))
        for step in range(2):
            if step == 1 and run_probe:
                python_rng, numpy_rng = random.getstate(), np.random.get_state()
                gradients = probe.probe_train_global_model(probe_epochs=1)
                assert set(gradients) == {"0.weight", "2.weight"}
                assert builder.parameters is params
                assert random.getstate() == python_rng
                np.testing.assert_equal(np.random.get_state(), numpy_rng)
            x, y = next(iter(loader))
            optimizer.zero_grad()
            loss(model(x), y).backward()
            optimizer.step()
        return model.state_dict(), optimizer.state_dict()

    torch.testing.assert_close(train(True), train(False), rtol=0, atol=0)


def main():
    console.out("Test optimizer from a yaml file")
    console.out("------------- Begin ---------------")
    test_from_file()
    console.out("------------- End -----------------\n")

    console.out("Test optimizer from config dict")
    console.out("------------- Begin ---------------")
    test_direct()
    console.out("------------- End -----------------")
    return

if __name__ == "__main__":
    main()
