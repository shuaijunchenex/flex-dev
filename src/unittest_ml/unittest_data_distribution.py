from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_data_process import DataDistribution
from flex.ml_utils import ConfigLoader


def test_standard_distribution():
    d1 = DataDistribution.use("mnist_lt")
    print(f"Standard name: mnist_lt: \n{d1}\n")

    d2 = DataDistribution.use("mnist_lt_one_label")
    print(f"Standard name: mnist_lt_one_label: \n{d2}\n")

    d3 = DataDistribution.use("mnist_data_volum_balance")
    print(f"Standard name: mnist_data_volum_balance: \n{d3}\n")
    return


def test_custom_distribution():

    config_file = './test_data/node_config_template_client.yaml'
    config_dict = ConfigLoader.load(config_file)

    DataDistribution.parse_config(config_dict)

    d = DataDistribution.get()
    print(f"Config use distribution:  \n{d}\n")

    return


def main():
    test_standard_distribution()
    test_custom_distribution()

    return

if __name__ == "__main__":
    main()
