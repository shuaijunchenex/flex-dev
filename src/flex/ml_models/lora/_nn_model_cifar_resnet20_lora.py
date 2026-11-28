from collections import OrderedDict

from torch import nn
from torchvision.models.resnet import BasicBlock

from ._nn_model_cifar_pretrained_cnn import NNModel_CifarPretrainedLoRA


def _resnet_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    shortcut = None
    if stride != 1:
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    return nn.Sequential(
        BasicBlock(in_channels, out_channels, stride, downsample=shortcut),
        BasicBlock(out_channels, out_channels),
        BasicBlock(out_channels, out_channels),
    )


class NNModel_CifarResNet20LoRA(NNModel_CifarPretrainedLoRA):
    classifier_name = "fc"

    @staticmethod
    def _build_backbone() -> nn.Module:
        return nn.Sequential(OrderedDict(
            conv1=nn.Conv2d(3, 16, 3, padding=1, bias=False),
            bn1=nn.BatchNorm2d(16),
            relu=nn.ReLU(inplace=True),
            layer1=_resnet_stage(16, 16, 1),
            layer2=_resnet_stage(16, 32, 2),
            layer3=_resnet_stage(32, 64, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(),
            fc=nn.Linear(64, 10),
        ))

    @staticmethod
    def _is_lora_target(name: str, layer: nn.Conv2d) -> bool:
        return name.startswith(("layer1.", "layer2.", "layer3.")) and name.endswith((".conv1", ".conv2"))
