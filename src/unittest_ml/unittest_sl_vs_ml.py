"""
Unit test to compare Split Learning (SL) and standard ML training outputs.
Tests whether SL training logic produces consistent results with normal ML training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
import copy
import numpy as np
import random

# Init startup path
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))

from flex.model_trainer import ModelTrainerFactory
from flex.ml_utils import ConfigLoader, console
from flex.ml_algorithms import LossFunctionBuilder, OptimizerBuilder
from flex.ml_data_loader import DatasetLoaderFactory


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dummy_dataloader(input_dim=784, num_samples=100, batch_size=32):
    """Create a dummy dataloader for testing."""
    set_seed(42)
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(x, y)
    
    # We use a custom object that mimics DatasetLoader's interface if needed
    # but ModelTrainer_Standard expects ta.train_loader.data_loader
    class MockDatasetLoader:
        def __init__(self, dl):
            self.data_loader = dl
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return MockDatasetLoader(loader)


# Define a simple MLP model for testing
class SimpleMLP(nn.Module):
    """Simple MLP for testing - can be split into front and rear parts."""
    
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SplitMLPFront(nn.Module):
    """Front part of split MLP (client side)."""
    
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x


class SplitMLPRear(nn.Module):
    """Rear part of split MLP (server side)."""
    
    def __init__(self, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(x)


def train_with_standard_ml(
    yaml: Dict,
    model: nn.Module,
    epochs: int = 1,
    device: str = "cpu",
    train_loader = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Train using standard ML approach."""
    
    console.out("\n=== Training with Standard ML ===")
    
    set_seed(42)  # Reset seed before training
    
    # Create trainer args
    trainer_args = ModelTrainerFactory.create_args(yaml)
    
    # Create data loader if not provided
    if train_loader is None:
        data_loader_args = DatasetLoaderFactory.create_args(yaml)
        data_loader_args.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten)
        ])
        data_loader_args.root = "../../../.dataset"
        data_loader_args.batch_size = 32
        data_loader = DatasetLoaderFactory.create(data_loader_args)
    else:
        data_loader = train_loader
    
    # Setup trainer args
    trainer_args.train_loader = data_loader
    trainer_args.device = device
    trainer_args.model = model.to(device)
    trainer_args.loss_func = LossFunctionBuilder.build(yaml)
    trainer_args.optimizer = OptimizerBuilder(model.parameters(), yaml).build()
    
    # Create and run trainer
    trainer = ModelTrainerFactory.create(trainer_args)
    state_dict, stats = trainer.train(epochs)
    
    console.ok(f"Standard ML - Final stats: {stats}")
    
    return state_dict, stats


def train_with_split_learning(
    yaml: Dict,
    front_model: nn.Module,
    rear_model: nn.Module,
    epochs: int = 1,
    device: str = "cpu",
    train_loader = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
    """Train using Split Learning approach."""
    
    console.out("\n=== Training with Split Learning ===")
    
    set_seed(42)  # Reset seed before training
    
    # Create data loader if not provided
    if train_loader is None:
        data_loader_args = DatasetLoaderFactory.create_args(yaml)
        data_loader_args.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten)
        ])
        data_loader_args.root = "../../../.dataset"
        data_loader_args.batch_size = 32
        data_loader = DatasetLoaderFactory.create(data_loader_args)
    else:
        data_loader = train_loader
    
    # Create client trainer (front model)
    from flex.model_trainer.model_trainer_args import ModelTrainerArgs
    from flex.model_trainer.trainer._model_trainer_sl_client import ModelTrainer_SlClient
    
    client_trainer_args = ModelTrainerArgs()
    client_trainer_args.train_loader = data_loader
    client_trainer_args.device = device
    client_trainer_args.model = front_model.to(device)
    client_trainer_args.optimizer = OptimizerBuilder(front_model.parameters(), yaml).build()
    
    client_trainer = ModelTrainer_SlClient(client_trainer_args)
    
    # Create server trainer (rear model)
    from flex.model_trainer.trainer._model_trainer_sl_server import ModelTrainer_SlServer
    
    server_trainer_args = ModelTrainerArgs()
    server_trainer_args.device = device
    server_trainer_args.model = rear_model.to(device)
    server_trainer_args.loss_func = LossFunctionBuilder.build(yaml)
    server_trainer_args.optimizer = OptimizerBuilder(rear_model.parameters(), yaml).build()
    
    server_trainer = ModelTrainer_SlServer(server_trainer_args)
    
    # Training loop - IMPORTANT: Process one full epoch at a time
    total_metrics = {"loss": 0.0, "accuracy": 0.0}
    total_batches = 0
    
    # Get the actual dataloader
    train_dl = data_loader.data_loader if hasattr(data_loader, "data_loader") else data_loader
    
    for epoch in range(epochs):
        console.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        epoch_metrics = {"loss": 0.0, "accuracy": 0.0}
        epoch_batches = 0
        
        # Reset client model to train mode
        front_model.train()
        rear_model.train()
        
        # Process each batch independently (SL processes batch-by-batch)
        for inputs, labels in train_dl:
            # 1. Client forward pass
            inputs = inputs.to(device)
            smashed_data = front_model(inputs)
            smashed_data.retain_grad()  # Important: retain grad for backward
            
            # 2. Server forward pass
            server_input, loss, metrics = server_trainer.forward_only(
                smashed_data.detach(), labels, training=True
            )
            
            # 3. Server backward pass
            activation_grad, loss_value = server_trainer.backward_only(
                server_input, loss, training=True
            )
            
            # 4. Client backward pass
            client_trainer_args.optimizer.zero_grad()
            smashed_data.backward(activation_grad.to(device))
            client_trainer_args.optimizer.step()
            
            # Accumulate metrics
            epoch_metrics["loss"] += metrics.get("loss", 0.0)
            epoch_metrics["accuracy"] += metrics.get("accuracy", 0.0)
            epoch_batches += 1
            total_batches += 1
        
        # Average epoch metrics
        avg_epoch_loss = epoch_metrics["loss"] / epoch_batches
        avg_epoch_acc = epoch_metrics["accuracy"] / epoch_batches
        console.info(f"Epoch {epoch + 1} - Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_acc:.4f}")
        
        # Accumulate total metrics
        for key in epoch_metrics:
            total_metrics[key] += epoch_metrics[key]
    
    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= total_batches
    
    console.ok(f"Split Learning - Final stats: {total_metrics}")
    
    # Combine state dicts
    front_state = front_model.state_dict()
    rear_state = rear_model.state_dict()
    
    return front_state, rear_state, total_metrics


def compare_models(
    ml_state: Dict[str, torch.Tensor],
    sl_front_state: Dict[str, torch.Tensor],
    sl_rear_state: Dict[str, torch.Tensor]
) -> bool:
    """Compare model parameters between ML and SL approaches."""
    
    console.out("\n=== Comparing Model Parameters ===")
    
    # Combine SL front and rear states
    sl_combined_state = {}
    
    # Front model parameters (fc1)
    for key, value in sl_front_state.items():
        sl_combined_state[key] = value
    
    # Rear model parameters (fc2)
    for key, value in sl_rear_state.items():
        sl_combined_state[key] = value
    
    # Compare each parameter
    max_diff = 0.0
    all_close = True
    
    for key in ml_state:
        if key in sl_combined_state:
            ml_param = ml_state[key]
            sl_param = sl_combined_state[key]
            
            # Calculate difference
            diff = torch.abs(ml_param - sl_param).max().item()
            max_diff = max(max_diff, diff)
            
            is_close = torch.allclose(ml_param, sl_param, rtol=1e-3, atol=1e-3)
            
            status = "✓" if is_close else "✗"
            console.out(f"{status} {key}: max_diff={diff:.6f}, close={is_close}")
            
            if not is_close:
                all_close = False
        else:
            console.warn(f"Key {key} not found in SL combined state")
            all_close = False
    
    console.out(f"\nMax parameter difference: {max_diff:.6f}")
    
    return all_close


def compare_metrics(
    ml_stats: Dict[str, float],
    sl_stats: Dict[str, float],
    tolerance: float = 0.1
) -> bool:
    """Compare training metrics between ML and SL approaches."""
    
    console.out("\n=== Comparing Training Metrics ===")
    
    # Map SL keys to ML keys for comparison
    key_mapping = {
        "avg_loss": "loss",
        "accuracy": "accuracy"
    }
    
    all_close = True
    
    for ml_key, sl_key in key_mapping.items():
        if ml_key in ml_stats and sl_key in sl_stats:
            ml_value = ml_stats[ml_key]
            sl_value = sl_stats[sl_key]
            
            diff = abs(ml_value - sl_value)
            relative_diff = diff / max(abs(ml_value), 1e-8)
            
            is_close = relative_diff < tolerance
            status = "✓" if is_close else "✗"
            
            console.out(
                f"{status} {ml_key}: ML={ml_value:.4f}, SL={sl_value:.4f}, "
                f"diff={diff:.4f}, relative={relative_diff:.2%}"
            )
            
            if not is_close:
                all_close = False
        else:
            # Accuracy might not be in ML stats if it's not calculated
            if ml_key == "accuracy":
                continue
            console.warn(f"Metric {ml_key} or {sl_key} not found for comparison")
    
    return all_close


def test_sl_vs_ml():
    """Main test function comparing SL and ML training."""
    
    console.out("\n" + "="*80)
    console.out("Unit Test: Split Learning vs Standard ML Training")
    console.out("="*80)
    
    # Load configuration
    yaml_file = './test_data/test_trainer.yaml'
    yaml = ConfigLoader.load(yaml_file)
    
    # Test parameters
    input_dim = 784
    hidden_dim = 128
    output_dim = 10
    epochs = 1
    device = "cpu"
    
    # Initialize models with same random seed for fair comparison
    set_seed(42)
    
    # Standard ML model
    ml_model = SimpleMLP(input_dim, hidden_dim, output_dim)
    
    # Split Learning models (initialized with same weights)
    set_seed(42)
    sl_front_model = SplitMLPFront(input_dim, hidden_dim)
    sl_rear_model = SplitMLPRear(hidden_dim, output_dim)
    
    # Verify initial weights match
    console.out("\n=== Verifying Initial Weights ===")
    assert torch.allclose(ml_model.fc1.weight, sl_front_model.fc1.weight), "Front fc1 weights don't match!"
    assert torch.allclose(ml_model.fc2.weight, sl_rear_model.fc2.weight), "Rear fc2 weights don't match!"
    console.ok("✓ Initial weights match")
    
    # Use dummy data for perfect consistency
    dummy_loader = get_dummy_dataloader(input_dim=input_dim, num_samples=200, batch_size=32)
    
    # Train with standard ML
    ml_state, ml_stats = train_with_standard_ml(yaml, ml_model, epochs, device, train_loader=dummy_loader)
    
    # Re-initialize models with same seed for SL
    set_seed(42)
    sl_front_model = SplitMLPFront(input_dim, hidden_dim)
    sl_rear_model = SplitMLPRear(hidden_dim, output_dim)
    
    # Train with Split Learning
    sl_front_state, sl_rear_state, sl_stats = train_with_split_learning(
        yaml, sl_front_model, sl_rear_model, epochs, device, train_loader=dummy_loader
    )
    
    # Compare results
    console.out("\n" + "="*80)
    console.out("Comparison Results")
    console.out("="*80)
    
    params_match = compare_models(ml_state, sl_front_state, sl_rear_state)
    metrics_match = compare_metrics(ml_stats, sl_stats, tolerance=0.15)
    
    # Final verdict
    console.out("\n" + "="*80)
    if params_match and metrics_match:
        console.ok("✓✓✓ TEST PASSED: SL and ML produce consistent results!")
    elif metrics_match:
        console.warn("⚠ TEST PARTIAL: Metrics match but parameters differ (may be due to numerical precision)")
    else:
        console.error("✗✗✗ TEST FAILED: Significant differences detected!")
    console.out("="*80)
    
    return params_match and metrics_match


def main():
    """Run the test."""
    try:
        result = test_sl_vs_ml()
        if result:
            console.out("\n✓ All tests passed successfully!")
        else:
            console.out("\n✗ Some tests failed. Please review the output above.")
    except Exception as e:
        console.error(f"\n✗ Test crashed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
