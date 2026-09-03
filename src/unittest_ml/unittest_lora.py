from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import console, ConfigLoader, ObjectMap
from flex.ml_algorithms import LoRAArgs

def main():

    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Load yaml file
    console.out(f"Test LoRA from {yaml_file_name}")
    console.out("------------- Begin ---------------")
    console.out(f"load yaml file: {yaml_file_name}")
    yaml = ConfigLoader.load(yaml_file_name)

    arg = LoRAArgs().with_config_dict(yaml)

    return

if __name__ == "__main__":
    main()
