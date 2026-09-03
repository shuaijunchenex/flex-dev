import unittest
import torch
import torch.nn as nn
from collections import OrderedDict

# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.fl_algorithms import FedAggregatorFactory, FedAggregatorArgs

# Dummy model: Simple MLP
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

def get_model_weights(model: nn.Module) -> OrderedDict:
    return OrderedDict({k: v.clone().detach() for k, v in model.state_dict().items()})

class TestFedAvgAggregator(unittest.TestCase):
    def test_fedavg_aggregation(self):
        # Set fixed random seed
        torch.manual_seed(42)

        # Create 3 dummy clients with different weights and volumes
        model1 = SimpleModel()
        model2 = SimpleModel()
        model3 = SimpleModel()

        for param in model2.parameters():
            param.data += 1.0  # offset model2

        for param in model3.parameters():
            param.data += 2.0  # offset model3

        # Extract state_dicts
        weights1 = get_model_weights(model1)
        weights2 = get_model_weights(model2)
        weights3 = get_model_weights(model3)

        # Simulate client data list: (state_dict, data_volume)
        client_data_list = [
            [weights1, 1],  # 1 sample
            [weights2, 2],  # 2 samples
            [weights3, 3],  # 3 samples
        ]

        # Aggregate
        # Args: 
        #  Create from factory
        #  aggregator_args = FedAggregatorFactory.create_args(yaml_dict)           
        # OR
        #  Create args object directly
        aggregator_args = FedAggregatorArgs()           
        aggregator = FedAggregatorFactory.create_aggregator(aggregator_args)
        aggregated_weights = aggregator.aggregate(client_data_list)

        # Manually compute expected result
        total = 1 + 2 + 3
        expected = OrderedDict()
        for key in weights1.keys():
            expected[key] = (
                weights1[key] * 1 +
                weights2[key] * 2 +
                weights3[key] * 3
            ) / total

        # Compare tensors
        for key in expected:
            self.assertTrue(torch.allclose(aggregated_weights[key], expected[key], atol=1e-6),
                            f"Mismatch in key {key}")

        print(aggregator.aggregated_weight)
        print(expected)

        print("[Test] FedAvg Aggregator passed.")
        

if __name__ == '__main__':
    unittest.main()
