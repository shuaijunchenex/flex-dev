from torch import nn
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5

from ._nn_model_cifar_pretrained_cnn import NNModel_CifarPretrainedLoRA


class NNModel_CifarShuffleNetV2LoRA(NNModel_CifarPretrainedLoRA):
    classifier_name = "fc"

    @staticmethod
    def _build_backbone() -> nn.Module:
        model = shufflenet_v2_x0_5(weights=None, num_classes=10)
        # The CIFAR release keeps the stem at full image resolution.
        model.conv1[0].stride = (1, 1)
        model.maxpool = nn.Identity()
        return model

    @staticmethod
    def _is_lora_target(name: str, layer: nn.Conv2d) -> bool:
        return layer.kernel_size == (1, 1) and layer.groups == 1
