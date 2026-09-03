import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel
from ...ml_algorithms.lora import MSLoRALinear, MSLoRAConv2d
from ...ml_utils import console
# from your_module.lora_layers import MSLoRAConv2d, MSLoRALinear


class NNModel_CifarLoRACNN(NNModel):
    """
    更深版本（无大FC）：
      [Block1] 3→32: Conv-BN-ReLU-Conv-BN-ReLU + MaxPool(2) + Dropout
      [Block2] 32→64: 同上
      [Block3] 64→128: 同上
      GAP(1x1) → Linear(128→10)  （输出 logits, 不做 softmax）
    CIFAR-10 输入: [N, 3, 32, 32]
    """
    def __init__(self):
        super().__init__()
        # 占位（与你现有代码风格一致）
        self.conv1a = self.bn1a = self.conv1b = self.bn1b = self.pool1 = self.do1 = None
        self.conv2a = self.bn2a = self.conv2b = self.bn2b = self.pool2 = self.do2 = None
        self.conv3a = self.bn3a = self.conv3b = self.bn3b = self.pool3 = self.do3 = None
        self.gap = None
        self.fc = None

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)

        # LoRA/正则配置
        rank_ratio      = float(getattr(args, "rank_ratio", 1))
        base_r_conv     = int(getattr(args, "lora_rank_conv", 64))
        r_conv          = int(max(1, round(base_r_conv * rank_ratio)))
        base_r_fc       = int(getattr(args, "lora_rank_fc", 16))
        r_fc            = int(max(1, round(base_r_fc * rank_ratio)))
        if (base_r_conv == 1 or base_r_fc == 1) and rank_ratio != 1.0:
            console.warn(
                f"[CIFAR LoRA CNN] rank_ratio={rank_ratio} is configured but "
                f"base lora_rank_conv={base_r_conv} / lora_rank_fc={base_r_fc}, "
                f"rank=1 will stay 1 regardless of rank_ratio. "
                f"Consider increasing lora_rank_conv / lora_rank_fc in the config."
            )
        alpha_conv    = int(getattr(args, "lora_alpha_conv", r_conv if r_conv > 0 else 1))
        alpha_fc      = int(getattr(args, "lora_alpha_fc",   r_fc   if r_fc   > 0 else 1))
        lora_dropout  = float(getattr(args, "lora_dropout", 0.0))
        p_dropout     = float(getattr(args, "dropout", 0.25))
        merge_weights = bool(getattr(args, "merge_weights", False))
        use_bias      = bool(getattr(args, "use_bias", True))

        # -------- Block 1: 3 -> 32 --------
        self.conv1a = MSLoRAConv2d(3, 32, kernel_size=3, padding=1,
                                   r=r_conv, lora_alpha=alpha_conv, lora_dropout=lora_dropout,
                                   merge_weights=merge_weights, bias=use_bias)
        self.bn1a   = nn.BatchNorm2d(32)
        self.conv1b = MSLoRAConv2d(32, 32, kernel_size=3, padding=1,
                                   r=r_conv, lora_alpha=alpha_conv, lora_dropout=lora_dropout,
                                   merge_weights=merge_weights, bias=use_bias)
        self.bn1b   = nn.BatchNorm2d(32)
        self.pool1  = nn.MaxPool2d(2)   # 32x32 -> 16x16
        self.do1    = nn.Dropout(p_dropout)

        # -------- Block 2: 32 -> 64 --------
        self.conv2a = MSLoRAConv2d(32, 64, kernel_size=3, padding=1,
                                   r=r_conv, lora_alpha=alpha_conv, lora_dropout=lora_dropout,
                                   merge_weights=merge_weights, bias=use_bias)
        self.bn2a   = nn.BatchNorm2d(64)
        self.conv2b = MSLoRAConv2d(64, 64, kernel_size=3, padding=1,
                                   r=r_conv, lora_alpha=alpha_conv, lora_dropout=lora_dropout,
                                   merge_weights=merge_weights, bias=use_bias)
        self.bn2b   = nn.BatchNorm2d(64)
        self.pool2  = nn.MaxPool2d(2)   # 16x16 -> 8x8
        self.do2    = nn.Dropout(p_dropout)

        # -------- Block 3: 64 -> 128 --------
        self.conv3a = MSLoRAConv2d(64, 128, kernel_size=3, padding=1,
                                   r=r_conv, lora_alpha=alpha_conv, lora_dropout=lora_dropout,
                                   merge_weights=merge_weights, bias=use_bias)
        self.bn3a   = nn.BatchNorm2d(128)
        self.conv3b = MSLoRAConv2d(128, 128, kernel_size=3, padding=1,
                                   r=r_conv, lora_alpha=alpha_conv, lora_dropout=lora_dropout,
                                   merge_weights=merge_weights, bias=use_bias)
        self.bn3b   = nn.BatchNorm2d(128)
        self.pool3  = nn.MaxPool2d(2)   # 8x8 -> 4x4
        self.do3    = nn.Dropout(p_dropout)

        # -------- Head: GAP + Linear(128->10) --------
        self.gap = nn.AdaptiveAvgPool2d(1)  # 4x4 -> 1x1
        self.fc  = MSLoRALinear(in_features=128, out_features=10,
                                r=r_fc, lora_alpha=alpha_fc, lora_dropout=lora_dropout,
                                fan_in_fan_out=False, merge_weights=merge_weights, bias=True)

        # Declare LoRA configuration for aggregation system
        self._lora_config = {
            "suffix_A": "lora_A",
            "suffix_B": "lora_B",
            "sp_suffix": "sp_aggregated",
            "has_non_lora_params": True,   # BatchNorm layers are non-LoRA
        }

        return self

    # override
    def forward(self, x):
        # Block1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x); x = self.do1(x)

        # Block2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x); x = self.do2(x)

        # Block3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x); x = self.do3(x)

        # Head
        x = self.gap(x)               # [N, 128, 1, 1]
        x = torch.flatten(x, 1)       # [N, 128]
        logits = self.fc(x)           # [N, 10]
        return logits

    # 可选：统一切换 LoRA 模式
    def set_lora_mode(self, mode: str):
        if mode not in ["standard", "lora_only", "lora_disabled", "scaling"]:
            raise ValueError(f"Unsupported lora_mode: {mode}")
        for m in [self.conv1a, self.conv1b, self.conv2a, self.conv2b, self.conv3a, self.conv3b, self.fc]:
            if hasattr(m, "lora_mode"):
                m.lora_mode = mode