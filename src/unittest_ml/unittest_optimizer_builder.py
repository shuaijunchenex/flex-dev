from __future__ import annotations
import torch


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import ConfigLoader, console
from flex.ml_algorithms import OptimizerBuilder

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
