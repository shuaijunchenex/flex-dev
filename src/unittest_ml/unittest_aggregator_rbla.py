import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from flex.fl_algorithms import FedAggregatorFactory, FedAggregatorArgs
from flex.ml_algorithms import MSLoRAConv2d, MSLoRALinear

from copy import deepcopy

class LoRACNN(nn.Module):
    def __init__(self, num_classes=10, r=4, lora_alpha=16, lora_dropout=0.1):
        super(LoRACNN, self).__init__()
        self.conv1 = MSLoRAConv2d(1, 32, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.conv2 = MSLoRAConv2d(32, 64, kernel_size=3, padding=1, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = MSLoRALinear(64 * 7 * 7, 128, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.fc2 = MSLoRALinear(128, num_classes, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def test_lora_cnn_aggregation_and_distribution():
    """RBLA NaN-padding 聚合正确性验证：用已知常量权重手动验算。

    设计：
      - Client 0: r=1, 所有 lora_A / lora_B 值全为 1.0
      - Client 1: r=2, 所有 lora_A / lora_B 值全为 2.0
      - Client 2: r=3, 所有 lora_A / lora_B 值全为 3.0
    - 非 LoRA float tensor 分别设为 10.0 / 20.0 / 30.0
      - 等权重 (各 1/3)

    NaN-aware 加权平均的预期（以 conv1.lora_A [r·k, C_in·k] = [r×3, 3] 为例）：
      padding 到 [9, 3]（max rows = 3×3 = 9）。
      - rows 0-1: 三个 client 都有 → (1+2+3)/3 = 2.0
      - rows 3-5: client 1 没有 (NaN) → (2+3)/2 = 2.5
      - rows 6-8: 只有 client 2 → 3.0 / 1 = 3.0

    lora_B 同理，按列维度（dim=1）做 NaN-padding。
    """
    # ── 创建 r=1,2,3 的模型 ──
    models = [LoRACNN(r=r) for r in [1, 2, 3]]

    # ── 手动设置所有权重为已知常量 ──
    state_dicts = []
    for i, model in enumerate(models):
        r = i + 1  # 1, 2, 3
        sd = model.state_dict()
        for k in sd:
            if "lora" in k:
                sd[k] = torch.full_like(sd[k], float(r))       # 全 r
            else:
                sd[k] = torch.full_like(sd[k], float(r * 10))  # 非 LoRA 全 10/20/30
        state_dicts.append(sd)

    # ── 聚合 ──
    agg_args = FedAggregatorArgs({"method": "rbla", "device": "cpu"})
    aggregator = FedAggregatorFactory.create_aggregator(agg_args)
    sample_nums = [1, 1, 1]  # 等权重
    client_data = [
        {"updated_weights": sd, "train_record": {"data_sample_num": n}}
        for sd, n in zip(state_dicts, sample_nums)
    ]
    global_sd = aggregator.aggregate(client_data)

    # ================================================================
    # 验算非 LoRA 层：rank 无关 tensor 直接 weighted average
    # ================================================================
    expected_non_lora = 20.0  # (10 + 20 + 30) / 3
    non_lora_checks = [
        ("conv1.weight", global_sd["conv1.weight"][0, 0, 0, 0].item()),
        ("conv1.bias", global_sd["conv1.bias"][0].item()),
        ("fc1.weight", global_sd["fc1.weight"][0, 0].item()),
        ("fc2.bias", global_sd["fc2.bias"][0].item()),
    ]
    for key, actual in non_lora_checks:
        assert abs(actual - expected_non_lora) < 1e-5, (
            f"{key}: expected weighted average {expected_non_lora}, got {actual}"
        )
    print(f"  ✅ non-LoRA tensors weighted average correct: {expected_non_lora}")

    # ================================================================
    # 逐层手动验算
    # ================================================================
    # 选一个代表性的 conv 层 (conv1) 和 fc 层 (fc1) 来检查
    for layer_prefix in ["conv1", "conv2", "fc1", "fc2"]:
        key_A = f"{layer_prefix}.lora_A"
        key_B = f"{layer_prefix}.lora_B"
        A_global = global_sd[key_A]   # 2D: [R_max, d_in]
        B_global = global_sd[key_B]   # 2D: [d_out, R_max]

        R_max = A_global.shape[0]     # 最大 rank（dim 0）

        # --- 验算 lora_A 按行 ---
        # rows 0..r1-1: 三个 client 都有 → (1+2+3)/3 = 2.0
        r1 = 1 * 3 if "conv" in layer_prefix else 1
        # Actually, r1 is the model's r value for that layer.
        # For conv layers: lora_A dim 0 = r * kernel_size. For fc: dim 0 = r.
        # Since all clients use the same kernel_size, we can just use the shape.
        # Let's use a simpler approach: check specific rows.

        # We know:
        # - Client 0 has r=1 → covered rows 0..(r_1-1) where r_1 = 1*k or 1
        # - Client 1 has r=2 → covered rows 0..(r_2-1)
        # - Client 2 has r=3 → covered rows 0..(r_3-1)

        # For a conv layer with k=3: r_1=3, r_2=6, r_3=9
        # For a fc layer: r_1=1, r_2=2, r_3=3

        if "conv" in layer_prefix:
            r_1, r_2, r_3 = 3, 6, 9
        else:
            r_1, r_2, r_3 = 1, 2, 3

        # --- lora_A checks ---
        # Row in [0, r_1): all 3 clients → 2.0
        a_val_low = A_global[0, 0].item()
        assert abs(a_val_low - 2.0) < 1e-5, (
            f"{key_A} row 0: expected 2.0, got {a_val_low}"
        )

        # Row in [r_1, r_2): clients 1,2 → 2.5
        if r_2 > r_1:
            a_val_mid = A_global[r_1, 0].item()
            assert abs(a_val_mid - 2.5) < 1e-5, (
                f"{key_A} row {r_1}: expected 2.5, got {a_val_mid}"
            )

        # Row in [r_2, r_3): only client 2 → 3.0
        if r_3 > r_2:
            a_val_hi = A_global[r_2, 0].item()
            assert abs(a_val_hi - 3.0) < 1e-5, (
                f"{key_A} row {r_2}: expected 3.0, got {a_val_hi}"
            )

        # --- lora_B checks ---
        # Col in [0, r_1): all 3 clients → 2.0
        b_val_low = B_global[0, 0].item()
        assert abs(b_val_low - 2.0) < 1e-5, (
            f"{key_B} col 0: expected 2.0, got {b_val_low}"
        )

        # Col in [r_1, r_2): clients 1,2 → 2.5
        if r_2 > r_1:
            b_val_mid = B_global[0, r_1].item()
            assert abs(b_val_mid - 2.5) < 1e-5, (
                f"{key_B} col {r_1}: expected 2.5, got {b_val_mid}"
            )

        # Col in [r_2, r_3): only client 2 → 3.0
        if r_3 > r_2:
            b_val_hi = B_global[0, r_2].item()
            assert abs(b_val_hi - 3.0) < 1e-5, (
                f"{key_B} col {r_2}: expected 3.0, got {b_val_hi}"
            )

        print(f"  ✅ {layer_prefix}: lora_A/B NaN-padding aggregation correct "
              f"(low={a_val_low}, mid={a_val_mid if r_2>r_1 else 'N/A'}, hi={a_val_hi if r_3>r_2 else 'N/A'})")

    # ================================================================
    # 验算 broadcast: 每个 client 拿回正确的 rank
    # ================================================================
    print("\n--- Broadcast verification ---")
    for i, model in enumerate(models):
        r = i + 1
        local_sd = model.state_dict()
        new_sd = aggregator.broadcast_lora_state_dict(global_sd, local_sd)

        for k in new_sd:
            if "lora_A" in k:
                expected_r = local_sd[k].shape[0]
                actual_r   = new_sd[k].shape[0]
                assert expected_r == actual_r, (
                    f"[Client {i}] {k} A rank mismatch: expected {expected_r}, got {actual_r}"
                )
                # 第一行第一列应该 = 2.0（三个 client 都有）
                assert abs(new_sd[k][0, 0].item() - 2.0) < 1e-5, (
                    f"[Client {i}] {k}[0,0] expected 2.0, got {new_sd[k][0,0].item()}"
                )
            if "lora_B" in k:
                expected_r = local_sd[k].shape[1]
                actual_r   = new_sd[k].shape[1]
                assert expected_r == actual_r, (
                    f"[Client {i}] {k} B rank mismatch: expected {expected_r}, got {actual_r}"
                )
                assert abs(new_sd[k][0, 0].item() - 2.0) < 1e-5, (
                    f"[Client {i}] {k}[0,0] expected 2.0, got {new_sd[k][0,0].item()}"
                )

        model.load_state_dict(new_sd, strict=False)
        print(f"  ✅ Client {i} (r={r}) broadcast verified: "
              f"lora_A[0,0]={new_sd['conv1.lora_A'][0,0].item():.1f}, "
              f"lora_B[0,0]={new_sd['conv1.lora_B'][0,0].item():.1f}")

    print("\n🎉 All RBLA NaN-padding aggregation & broadcast checks passed.\n")

# ==== 执行测试 ====
test_lora_cnn_aggregation_and_distribution()
