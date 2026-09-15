from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import ConfigLoader, console


def test_config_loader():
    json_file_name = './test_data/test_config.json'
    yaml_file_name = './test_data/test_config.yaml'
    yml_file_name = './test_data/test_config.yml'

    # Load JSON file
    console.out(f"Test load json file: {json_file_name}")
    console.out("------------- Begin ---------------")
    json_data = ConfigLoader.load(json_file_name)
    console.info("Loaded JSON data:")
    console.info(json_data)
    console.out("------------- End -----------------\n")

    # Load YAML file
    console.out(f"Test load yaml(*.yaml) file: {yaml_file_name}")
    console.out("------------- Begin ---------------")
    yaml_data = ConfigLoader.load(yaml_file_name)
    console.info("Loaded YAML data:")
    console.info(yaml_data)
    console.out("------------- End -----------------\n")

    console.out(f"Test load yaml(*.yml) file: {yml_file_name}")
    console.out("------------- Begin ---------------")
    yaml_data = ConfigLoader.load(yml_file_name)
    console.info("Loaded YAML data:")
    console.info(yaml_data)
    console.out("------------- End -----------------\n")
    return


def main():
    test_config_loader()
    return


if __name__ == "__main__":
    main()
