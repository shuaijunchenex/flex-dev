from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.fed_runner import FedRunner
from flex.ml_utils import ConfigLoader, String

def test_fed_runner_parser():
    # Load config from yaml file
    config = ConfigLoader.load("./test_data/fed_runner_template.yaml")

    fed_runner = FedRunner().with_yaml(config)
    fed_runner.prepare()
    fed_runner.run()
    return


if __name__ == "__main__":
    test_fed_runner_parser()
