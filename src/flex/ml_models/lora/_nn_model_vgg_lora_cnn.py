import torch
import torch.nn as nn

from .. import AbstractNNModel, NNModelArgs, NNModel
from ...ml_algorithms.lora import MSLoRALinear, MSLoRAConv2d

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
              512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_LoRA(nn.Module):
    def __init__(
        self,
        vgg_name: str = 'VGG11',
        num_classes: int = 10,
        # LoRA 配置
        r_conv: int = 8,
        r_fc: int = 16,
        alpha_conv: int | None = None,
        alpha_fc: int | None = None,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        use_bias: bool = True,
        # 其它
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if alpha_conv is None:
            alpha_conv = r_conv if r_conv > 0 else 1
        if alpha_fc is None:
            alpha_fc = r_fc if r_fc > 0 else 1

        self.features = self._make_layers(
            cfg[vgg_name],
            r_conv=r_conv,
            alpha_conv=alpha_conv,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            use_bias=use_bias,
            use_batchnorm=use_batchnorm,
        )
        # CIFAR-10 的 VGG 简化头：全局 5 次池化后通常是 1x1，取 512 通道线性分类
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)  # 与你原版一致；也可以换 AdaptiveAvgPool2d(1)
        self.classifier = MSLoRALinear(
            in_features=512,
            out_features=num_classes,
            r=r_fc,
            lora_alpha=alpha_fc,
            lora_dropout=lora_dropout,
            fan_in_fan_out=False,
            merge_weights=merge_weights,
            bias=True
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)  # logits（训练用 CrossEntropyLoss，不做 softmax）
        return out

    def _make_layers(
        self,
        cfg_list,
        *,
        r_conv: int,
        alpha_conv: int,
        lora_dropout: float,
        merge_weights: bool,
        use_bias: bool,
        use_batchnorm: bool,
    ):
        layers = []
        in_channels = 3
        for x in cfg_list:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = MSLoRAConv2d(
                    in_channels=in_channels,
                    out_channels=x,
                    kernel_size=3,
                    padding=1,              # VGG 用 same padding
                    r=r_conv,
                    lora_alpha=alpha_conv,
                    lora_dropout=lora_dropout,
                    merge_weights=merge_weights,
                    bias=use_bias
                )
                if use_batchnorm:
                    layers += [conv, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    # 可选：统一切换 LoRA 模式（如果你的层实现有 lora_mode）
    def set_lora_mode(self, mode: str):
        for m in self.modules():
            if hasattr(m, "lora_mode"):
                m.lora_mode = mode


# ---------------------------------------------------------------------------
# VGG9 LoRA — compact VGG-9 with LoRA (6 Conv + 3 FC)
# Architecture matches the non-LoRA VGG9 in _nn_model_vgg.py:
#   Block-1: Conv(3→64)  ×2 + MaxPool
#   Block-2: Conv(64→128) ×2 + MaxPool
#   Block-3: Conv(128→256)×2 + MaxPool
#   AdaptiveAvgPool2d(4,4) → 4096
#   FC: 4096 → 512 → 512 → num_classes
# ---------------------------------------------------------------------------
class VGG9_LoRA(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        r_conv: int = 8,
        r_fc: int = 16,
        alpha_conv: int | None = None,
        alpha_fc: int | None = None,
        lora_dropout: float = 0.0,
        dropout: float = 0.5,
        merge_weights: bool = False,
        use_bias: bool = True,
    ):
        super().__init__()
        if alpha_conv is None:
            alpha_conv = r_conv if r_conv > 0 else 1
        if alpha_fc is None:
            alpha_fc = r_fc if r_fc > 0 else 1

        def _conv(in_ch, out_ch):
            return MSLoRAConv2d(in_ch, out_ch, kernel_size=3, padding=1,
                                r=r_conv, lora_alpha=alpha_conv, lora_dropout=lora_dropout,
                                merge_weights=merge_weights, bias=use_bias)

        self.features = nn.Sequential(
            _conv(3, 64),   nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            _conv(64, 64),  nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            _conv(64, 128), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            _conv(128, 128),nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            _conv(128, 256),nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            _conv(256, 256),nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        def _linear(in_f, out_f):
            return MSLoRALinear(in_f, out_f, r=r_fc, lora_alpha=alpha_fc,
                                lora_dropout=lora_dropout, fan_in_fan_out=False,
                                merge_weights=merge_weights, bias=True)

        self.classifier = nn.Sequential(
            _linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            _linear(512, 512),         nn.ReLU(inplace=True), nn.Dropout(dropout),
            _linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_lora_mode(self, mode: str):
        for m in self.modules():
            if hasattr(m, "lora_mode"):
                m.lora_mode = mode


# ---------------------------------------------------------------------------
# NNModel wrappers for factory registration
# ---------------------------------------------------------------------------
class NNModel_VGG11_LoRA(NNModel):
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        return VGG_LoRA(
            vgg_name='VGG11',
            num_classes=getattr(args, "num_classes", 10) or 10,
            r_conv=getattr(args, "lora_rank_conv", 8) or 8,
            r_fc=getattr(args, "lora_rank_fc", 16) or 16,
            alpha_conv=getattr(args, "lora_alpha_conv", None),
            alpha_fc=getattr(args, "lora_alpha_fc", None),
            lora_dropout=getattr(args, "lora_dropout", 0.0) or 0.0,
            merge_weights=getattr(args, "merge_weights", False),
            use_bias=getattr(args, "use_bias", True),
        )


class NNModel_VGG9_LoRA(NNModel):
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        rank_ratio_value = getattr(args, "rank_ratio", 1.0)
        rank_ratio = 1.0 if rank_ratio_value is None else float(rank_ratio_value)
        if rank_ratio <= 0.0:
            raise ValueError("rank_ratio must be positive for VGG9 LoRA")

        base_r_conv = int(getattr(args, "lora_rank_conv", 8) or 8)
        base_r_fc = int(getattr(args, "lora_rank_fc", 16) or 16)
        r_conv = max(1, round(base_r_conv * rank_ratio))
        r_fc = max(1, round(base_r_fc * rank_ratio))

        return VGG9_LoRA(
            num_classes=getattr(args, "num_classes", 10) or 10,
            r_conv=r_conv,
            r_fc=r_fc,
            alpha_conv=getattr(args, "lora_alpha_conv", None),
            alpha_fc=getattr(args, "lora_alpha_fc", None),
            lora_dropout=getattr(args, "lora_dropout", 0.0) or 0.0,
            dropout=getattr(args, "dropout", 0.5) or 0.5,
            merge_weights=getattr(args, "merge_weights", False),
            use_bias=getattr(args, "use_bias", True),
        )


# --------- quick test ----------
def test():
    net = VGG_LoRA('VGG11', num_classes=10, r_conv=8, r_fc=16, lora_dropout=0.05)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)  # torch.Size([2, 10])

if __name__ == "__main__":
    test()
