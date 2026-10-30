from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_algorithms import LossFunctionBuilder
from flex.ml_utils import console, ConfigLoader



def test_loss_function_builder():
    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Form yaml file
    console.out(f"Test build from yaml file: {yaml_file_name}")
    console.out("------------- Begin ---------------")
    yaml = ConfigLoader.load(yaml_file_name)

    console.info(yaml["loss_func"])
    loss_func = LossFunctionBuilder.build(yaml["loss_func"])
    console.info(loss_func)
    console.out("------------- End -----------------\n")

    # Form config dictionary directly
    console.out(f"Test build from dictionary")
    console.out("------------- Begin ---------------")

    yaml_dict = { "loss_func": {
        "type": "mseloss",
        "reduction" : "mean",
        "weight": None}}

    console.info(yaml_dict)
    loss_func = LossFunctionBuilder.build(yaml_dict["loss_func"])
    console.info(loss_func)
    console.out("------------- End -----------------\n")

    return


def main():
    test_loss_function_builder()

    return

if __name__ == "__main__":
    main()
