
# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_models import NNModelFactory
from flex.ml_utils import ConfigLoader


def main():

    # Load config from yaml file
    config = ConfigLoader.load("./test_data/node_config_template_client.yaml")

    #create args
    args = NNModelFactory.create_args(config)
    print(args)

    nn_model = NNModelFactory.create(args)

    print(nn_model)


if __name__ == "__main__":
    main()
