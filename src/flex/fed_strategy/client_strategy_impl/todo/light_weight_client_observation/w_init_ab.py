import copy

from train_strategy.client_strategy.base_client_strategy import ClientStrategy
from trainer.standard_model_trainer import StandardModelTrainer
from tools.optimizer_builder import OptimizerBuilder
from tools.model_utils import ModelUtils
from lora.lora_implementation.lora_utils import LoRAUtils
from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter

class WInitAbClientTrainingStrategy(ClientStrategy):
    '''
    Light weight observation method for Weight + SVD(AB)

    Forward propagation is W + AB.

    AB is generated based on original weight W using SVD decomposition.

    '''
    def __init__(self, client):
        self.client = client

    def run_local_training(self):
        print(f"\n Training Client [{self.client.node_id}] ...\n")

        updated_weights, train_record = self.local_training()

        data_pack = {"node_id": self.client.node_id, "updated_weights": updated_weights, "train_record": train_record, "data_sample_num": len(self.client.args.train_data.dataset)}

        return data_pack
    
    def run_observation(self):
        print(f"\n Observation Client [{self.client.node_id}] ...\n")

        updated_weights, train_record = self.observation()

        data_pack = {"node_id": self.client.node_id, "train_record": train_record, "data_sample_num": len(self.client.args.train_data.dataset)}

        return data_pack

    def observation(self):
        '''
        For light-weight client observation training, we use the local LoRA model.
        '''

        # deepcopy global model for training
        observation_model = copy.deepcopy(self.client.args.local_lora_model)

        # set lora mode
        LoRAUtils.set_lora_mode_for_model(observation_model, 'standard')

        # set lora wbab
        LoRAModelWeightAdapter.set_wbab(observation_model, self.client.args.local_model_wbab)

        # clear gradients
        ModelUtils.clear_model_grads(observation_model)

        trainable_params = [p for p in observation_model.parameters() if p.requires_grad]

        optimizer = OptimizerBuilder(trainable_params, self.client.args.optimizer).optimizer

        # Initialize the model trainer
        self.trainer = StandardModelTrainer(observation_model,
                                    optimizer, #torch.optim.SGD(trainable_params, lr=0.01),
                                    self.client.args.loss_func,
                                    self.client.args.train_data)

        # Call the trainer for local training
        updated_weights, train_record = self.trainer.observe(int(self.client.args.local_epochs))

        return copy.deepcopy(updated_weights), train_record
    
    def local_training(self):
        # Correct
        # Set global weight

        train_model = copy.deepcopy(self.client.args.local_model)

        train_model.load_state_dict(self.client.args.global_weight)

        # clear gradients
        ModelUtils.clear_model_grads(train_model)

        trainable_params = [p for p in train_model.parameters() if p.requires_grad]

        optimizer = OptimizerBuilder(trainable_params, self.client.args.optimizer).optimizer

        # Initialize the model trainer
        self.trainer = StandardModelTrainer(train_model,#self.client.args.local_model,
                                    optimizer, #torch.optim.SGD(trainable_params, lr=0.01),
                                    self.client.args.loss_func,
                                    self.client.args.train_data)
        
        # Call the trainer for local training
        updated_weights, train_record = self.trainer.train(self.client.args.local_epochs)

        # Update model weights
        self.client.update_weights(updated_weights)

        self.client.args.local_model.load_state_dict(updated_weights)

        return copy.deepcopy(updated_weights), train_record