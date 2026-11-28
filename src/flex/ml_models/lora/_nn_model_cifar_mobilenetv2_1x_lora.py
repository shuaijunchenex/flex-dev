from torch import nn
from torchvision.models.mobilenetv2 import MobileNetV2

from ._nn_model_cifar_pretrained_cnn import NNModel_CifarPretrainedLoRA


class NNModel_CifarMobileNetV2LoRA(NNModel_CifarPretrainedLoRA):
    classifier_name = "classifier"

    @staticmethod
    def _build_backbone() -> nn.Module:
        # The CIFAR release keeps full resolution at the stem and the 24-channel stage.
        model = MobileNetV2(num_classes=10, inverted_residual_setting=[
            [1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 2], [6, 64, 4, 2],
            [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ])
        model.features[0][0].stride = (1, 1)
        return model

    @staticmethod
    def _is_lora_target(name: str, layer: nn.Conv2d) -> bool:
        return layer.kernel_size == (1, 1) and layer.groups == 1
