from typing import Dict, Any

class MetricCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.running_loss_batch_mean_sum: float = 0.0
        self.total_batch: int = 0
        self.running_loss_sample_weighted_sum: float = 0.0
        self.total_samples: int = 0

    def update(self, loss: float, batch_size: int):
        self.total_batch += 1
        self.running_loss_batch_mean_sum += loss
        self.running_loss_sample_weighted_sum += loss * batch_size
        self.total_samples += batch_size

    @property
    def avg_loss(self) -> float:
        return self.running_loss_batch_mean_sum / max(self.total_batch, 1)

    @property
    def keras_loss(self) -> float:
        return self.running_loss_sample_weighted_sum / max(self.total_samples, 1)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "avg_loss": self.avg_loss,
            "running_loss_sum": self.running_loss_batch_mean_sum,
            "num_batches": self.total_batch,
            "keras_loss": self.keras_loss,
            "keras_running_loss_weighted_sum": self.running_loss_sample_weighted_sum,
            "num_samples": self.total_samples,
        }
