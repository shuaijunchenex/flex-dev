from abc import ABC, abstractmethod

class WeightHandler(ABC):
    @abstractmethod
    def set_weights(self, model, weights):
        """将权重应用到模型上"""
        pass

    @abstractmethod
    def get_weights(self, model):
        """从模型中获取权重"""
        pass

class StandardWeightHandler(WeightHandler):
    def set_weights(self, model, weights):
        # 直接调用模型设置权重的方法
        model.load_state_dict(weights)
        
    def get_weights(self, model):
        return model.state_dict()
    
class LoRAWeightHandler(WeightHandler):
    def __init__(self, lora_settings=None):
        self.lora_settings = lora_settings

    def set_weights(self, model, weights):
        #TODO
        pass
    
    def get_weights(self, model):
        #TODO
        pass

# # 使用标准权重处理器
# std_handler = StandardWeightHandler()
# client_std = ClientNode(
#     node_id="client_1",
#     config=config,
#     model=model,
#     weight_handler=std_handler,
#     optimizer=optimizer,
#     loss_func=loss_func,
#     train_data=train_data,
#     test_data=test_data
# )
# client_std.run()  # 执行本地训练

# # 使用 LoRA 权重处理器
# lora_handler = LoRAWeightHandler(lora_settings={"some_param": 123})
# client_lora = ClientNode(
#     node_id="client_2",
#     config=config,
#     model=model,
#     weight_handler=lora_handler,
#     optimizer=optimizer,
#     loss_func=loss_func,
#     train_data=train_data,
#     test_data=test_data
# )
# client_lora.run()  # 执行本地训练

# # 如果需要同时使用原始权重和 LoRA 的 AB 矩阵，可以使用复合处理器
# composite_handler = CompositeWeightHandler(standard_handler=std_handler, lora_handler=lora_handler)
# client_composite = ClientNode(
#     node_id="client_3",
#     config=config,
#     model=model,
#     weight_handler=composite_handler,
#     optimizer=optimizer,
#     loss_func=loss_func,
#     train_data=train_data,
#     test_data=test_data
# )
# client_composite.run()  # 执行本地训练