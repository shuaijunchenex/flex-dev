import torch
import torch.nn as nn
import copy
import torch.nn.init as init
import math

from . import LoRALinear
from ..model_extractor import ModelExtractor

class MatrixApproximator:
    def __init__(self, base_model, use_sqrt = True, rank = 8, device = "cpu"):
        self.base_model = base_model
        self.use_sqrt = use_sqrt
        self.rank = rank
        self.device = device

    def simple_lora_model_generator(self, a_init_method='xavier', b_init_method='zeros'):
        model_copy = copy.deepcopy(self.base_model)
        self._replace_module_with_init_lora(model_copy, a_init_method, b_init_method)

        extractor = ModelExtractor()
        wbab = extractor.extract_layers(model_copy)
        return model_copy, wbab

    def approximate_lora_model(self):
        model_copy = copy.deepcopy(self.base_model)
        self._replace_module_with_approximation(model_copy)

        extractor = ModelExtractor()
        wbab = extractor.extract_layers(model_copy)
        return model_copy, wbab

    def _init_tensor(self, tensor, method):
        if method == "xavier":
            init.xavier_uniform_(tensor)
        elif method == "kaiming":
            init.kaiming_uniform_(tensor, a=math.sqrt(5))
        elif method == "normal":
            init.normal_(tensor, mean=0.0, std=0.02)
        elif method == "zeros":
            init.zeros_(tensor)
        elif method == "ones":
            init.ones_(tensor)
        else:
            raise ValueError(f"Unsupported init method: {method}")

    def _replace_module_with_init_lora(self, module, a_init_method, b_init_method):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                W = child.weight.data.to(self.device)
                bias_flag = child.bias is not None

                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=self.rank,
                    use_bias=bias_flag,
                    pretrained_weight=W,
                    device=self.device
                )

                with torch.no_grad():
                    self._init_tensor(lora_layer.lora_A, a_init_method)
                    self._init_tensor(lora_layer.lora_B, b_init_method)
                    if bias_flag:
                        lora_layer.bias.copy_(child.bias.data)    # TODO: pylance?

                setattr(module, name, lora_layer)
            else:
                self._replace_module_with_init_lora(child, a_init_method, b_init_method)

    def _replace_module_with_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                W = child.weight.data.to(self.device)
                bias_flag = child.bias is not None

                A, B = (
                    self.sqrt_approximation(W, self.rank)
                    if self.use_sqrt else
                    self.regular_approximation(W, self.rank)
                )

                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=self.rank,
                    use_bias=bias_flag,
                    pretrained_weight=W,
                    device=self.device,
                    scaling = 1
                )

                with torch.no_grad():
                    lora_layer.lora_A.copy_(A)
                    lora_layer.lora_B.copy_(B)
                    if bias_flag:
                        lora_layer.bias.copy_(child.bias.data)     # TODO: pylance?

                setattr(module, name, lora_layer)
            else:
                self._replace_module_with_approximation(child)


    @staticmethod
    def sqrt_approximation(W: torch.Tensor, rank: int):
        """
        Decomposes the matrix W into two matrices A and B such that A @ B approximates W,
        using truncated SVD.

        Args:
            W (torch.Tensor): The original matrix of shape (m, n).
            rank (int): The target rank for approximation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                A (m, rank), B (rank, n) such that W ≈ A @ B
        """
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh is V^T

        # Truncate to rank
        U_r = U[:, :rank]                    # (m, r)
        S_r = S[:rank]                       # (r,)
        Vh_r = Vh[:rank, :]                  # (r, n)

        # Compute square root of singular values
        S_root = torch.sqrt(S_r)            # (r,)

        # Construct A and B
        A = U_r @ torch.diag(S_root)        # (m, r)
        B = torch.diag(S_root) @ Vh_r       # (r, n)

        return A, B

    @staticmethod
    def regular_approximation(W: torch.Tensor, rank: int):
        """
        Decomposes the matrix W into two matrices A and B such that A @ B approximates W,
        using truncated SVD.

        Args:
            W (torch.Tensor): The original matrix of shape (m, n).
            rank (int): The target rank for approximation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                A (m, rank), B (rank, n) such that W ≈ A @ B
        """
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh is V^T

        # Truncate to rank
        U_r = U[:, :rank]             # (m, r)
        S_r = S[:rank]                # (r,)
        Vh_r = Vh[:rank, :]           # (r, n)

        # Fix: Turn S into a diagonal matrix
        S_diag = torch.diag(S_r)     # (r, r)

        # Construct A and B
        A = U_r @ S_diag             # (m, r)
        B = Vh_r                     # (r, n)

        return A, B

# def simple_lora_model_generator(self, a_init_method = 'xavier', b_init_method = 'zeros'):
    #         def init_tensor(tensor, method):
    #             if method == "xavier":
    #                 init.xavier_uniform_(tensor)
    #             elif method == "kaiming":
    #                 init.kaiming_uniform_(tensor, a=math.sqrt(5))
    #             elif method == "normal":
    #                 init.normal_(tensor, mean=0.0, std=0.02)
    #             elif method == "zeros":
    #                 init.zeros_(tensor)
    #             elif method == "ones":
    #                 init.ones_(tensor)
    #             else:
    #                 raise ValueError(f"Unsupported init method: {method}")

    #         def replace_module(module):
    #             for name, child in module.named_children():
    #                 if isinstance(child, nn.Linear):
    #                     W = child.weight.data.to(self.device)
    #                     bias_flag = child.bias is not None

    #                     lora_layer = LoRALinear(
    #                         in_features=child.in_features,
    #                         out_features=child.out_features,
    #                         rank=self.rank,
    #                         use_bias=bias_flag,
    #                         pretrained_weight=W,
    #                         device=self.device
    #                     )

    #                     with torch.no_grad():
    #                         init_tensor(lora_layer.lora_A, a_init_method)
    #                         init_tensor(lora_layer.lora_B, b_init_method)

    #                         if bias_flag:
    #                             lora_layer.bias.copy_(child.bias.data)

    #                     setattr(module, name, lora_layer)
    #                 else:
    #                     replace_module(child)

    #         model_copy = copy.deepcopy(self.base_model)
    #         replace_module(model_copy)

    #         return model_copy

    # def approximate_lora_model(self):
    #     def replace_module(module):
    #         for name, child in module.named_children():
    #             if isinstance(child, nn.Linear):
    #                 W = child.weight.data.to(self.device)
    #                 bias_flag = child.bias is not None

    #                 # SVD decomposition
    #                 A, B = (
    #                     self.sqrt_approximation(W, self.rank)
    #                     if self.use_sqrt else
    #                     self.regular_approximation(W, self.rank)
    #                 )

    #                 lora_layer = LoRALinear(
    #                     in_features=child.in_features,
    #                     out_features=child.out_features,
    #                     rank=self.rank,
    #                     use_bias=bias_flag,
    #                     pretrained_weight=W,
    #                     device=self.device
    #                 )

    #                 with torch.no_grad():
    #                     lora_layer.lora_A.copy_(A)
    #                     lora_layer.lora_B.copy_(B)
    #                     if bias_flag:
    #                         lora_layer.bias.copy_(child.bias.data)

    #                 setattr(module, name, lora_layer)
    #             else:
    #                 replace_module(child)

    #     model_copy = copy.deepcopy(self.base_model)
    #     replace_module(model_copy)
    #     return model_copy
