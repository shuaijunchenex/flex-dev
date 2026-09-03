from __future__ import annotations
import torch
import torch.nn as nn


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os

from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_algorithms import MatrixApproximator, LoRALinear, ModelExtractor

# Test: approximate a full model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(20, 10)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

def test_matrix_approximator():
    torch.manual_seed(42)

    # Test: matrix-level approximation
    m, n = 64, 128
    rank = 16
    W = torch.randn(m, n)

    A_sqrt, B_sqrt = MatrixApproximator.sqrt_approximation(W, rank)
    W_sqrt_approx = A_sqrt @ B_sqrt
    sqrt_error = torch.norm(W - W_sqrt_approx) / torch.norm(W)

    A_reg, B_reg = MatrixApproximator.regular_approximation(W, rank)
    W_reg_approx = A_reg @ B_reg
    reg_error = torch.norm(W - W_reg_approx) / torch.norm(W)

    print("Original matrix shape:", W.shape)
    print(f"Target rank: {rank}")
    print(f"[sqrt_approximation] Relative Frobenius Error: {sqrt_error:.6f}")
    print(f"[regular_approximation] Relative Frobenius Error: {reg_error:.6f}")
    assert A_sqrt.shape == (m, rank)
    assert B_sqrt.shape == (rank, n)
    assert A_reg.shape == (m, rank)
    assert B_reg.shape == (rank, n)
    print("Matrix shape assertions passed.")

    model = DummyModel()
    approximator = MatrixApproximator(model, rank=4, use_sqrt=True)
    lora_model, w_b_AB = approximator.approximate_lora_model()

    #w_b_AB = ModelExtractor().extract_layers(lora_model)

    # Verify that all linear layers have been replaced
    linear_count = sum(1 for m in lora_model.modules() if isinstance(m, nn.Linear))
    lora_count = sum(1 for m in lora_model.modules() if isinstance(m, LoRALinear))
    print(f"Replaced {lora_count} Linear layers with LoRALinear (expected 2).")
    assert linear_count == 0 and lora_count == 2
    print("Model layer replacement verified.")


def main():
    test_matrix_approximator()
    return

if __name__ == "__main__":
    main()
