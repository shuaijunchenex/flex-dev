import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, scaling: float = 0.5, use_bias: bool = True, pretrained_weight: torch.Tensor = None, lora_mode: str = "standard", device: str = "cpu"):
        """
        Linear layer with Low-Rank Adaptation (LoRA).

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            rank (int): Rank of the low-rank decomposition.
            alpha (float): Scaling factor for the low-rank update.
            use_bias (bool): If True, includes a bias parameter.
            pretrained_weight (Optional[torch.Tensor]): If provided, initializes self.weight with this tensor.
            lora_mode (str): Determines the LoRA inference mode. Options: "standard", "alternate", etc.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        self.lora_mode = lora_mode

        # Initialize weight matrix
        if pretrained_weight != None:
            # If pretrained weight is provided, use it directly (set requires_grad=False to freeze)
            self.weight = nn.Parameter(pretrained_weight, requires_grad=False)
            self.weight.requires_grad = False
        else:
            # Xavier Uniform initialization for weight matrix
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.zeros_(self.weight)  # Xavier Uniform initialization
            self.weight.requires_grad = False  # Ensure weight is not trainable

        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(out_features)) if use_bias else None
        self.bias.requires_grad = False  # Ensure bias is not trainable

        # LoRA trainable parameters with Xavier initialization
        self.lora_A = nn.Parameter(torch.empty(out_features, rank).to(device))
        self.lora_B = nn.Parameter(torch.empty(rank, in_features).to(device))
        nn.init.xavier_uniform_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    def set_lora_mode(self, mode: str):
        if mode not in ["standard", "lora_only", "lora_disabled", "scaling"]:
            raise ValueError(f"Unsupported lora_mode: {mode}")
        self.lora_mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LoRALinear layer."""
        
        # Compute low-rank update           
        delta_weight = torch.matmul(self.lora_A, self.lora_B)
        
        if self.lora_mode == "standard":
            effective_weight = self.weight + delta_weight
        elif self.lora_mode == "lora_only":
            effective_weight = delta_weight
        elif self.lora_mode == "lora_disabled":
            effective_weight = self.weight
        elif self.lora_mode == "scaling":
            effective_weight = (1 - self.scaling) * self.weight + self.scaling * delta_weight
        else:
            raise ValueError(f"Unsupported lora_mode: {self.lora_mode}")
        
        return F.linear(x, effective_weight, self.bias)

