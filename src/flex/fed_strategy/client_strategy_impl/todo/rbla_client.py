import sys
sys.path.insert(0, '')
import torch
import copy

from train_strategy.client_strategy.base_client_strategy import ClientStrategy
from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
from model_extractor.advanced_model_extractor import AdvancedModelExtractor
from trainer.standard_model_trainer import StandardModelTrainer
from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
from tools.optimizer_builder import OptimizerBuilder
from tools.model_utils import ModelUtils

class RBLAClientTrainingStrategy(ClientStrategy):
    def __init__(self, client):
        self.client = client

    def run_local_training(self):
        print(f"\nTraining Client [{self.client.node_id}] ...\n")

        updated_weights, train_record = self.local_training()

        data_pack = {"node_id": self.client.node_id, "updated_weights": updated_weights, "train_record": train_record, "data_sample_num": len(self.client.args.train_data.dataset)}

        #LoRAModelWeightAdapter.apply_weights_to_model(self.client.args.local_model, updated_weights)

        return data_pack
    
    def local_training(self):
        # self.client.args.local_model.load_state_dict(current_weight)
        # LoRAModelWeightAdapter.apply_weights_to_model(self.client.args.local_model, self.client.args.local_model_WbAB)
        
        train_model = copy.deepcopy(self.client.args.local_lora_model)

        LoRAModelWeightAdapter.apply_weights_to_model(train_model, self.client.args.global_wbab)

        # clear gradients
        ModelUtils.clear_model_grads(train_model)

        trainable_params = [p for p in train_model.parameters() if p.requires_grad]

        optimizer = OptimizerBuilder(trainable_params, self.client.args.optimizer).optimizer

        # Initialize the model trainer
        self.trainer = StandardModelTrainer(train_model,
                                    optimizer, #torch.optim.SGD(trainable_params, lr=0.01),
                                    self.client.args.loss_func,
                                    self.client.args.train_data)

        # Call the trainer for local training
        _ , train_record = self.trainer.train(self.client.args.local_epochs)

        # Extract the updated weights
        updated_weights = self.trainer.extract_WbAB()

        self.client.args.local_wbab = updated_weights

        return updated_weights, train_record