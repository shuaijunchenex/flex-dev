"""RBLA analysis runner that records client identity and saves a final checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

from ....ml_utils import console
from ....ml_utils.model_utils import ModelUtils
from .._rbla_runner_strategy import RblaRunnerStrategy


class RblaSupportAnalysisRunnerStrategy(RblaRunnerStrategy):
    def simulate_client_local_training_process(self, participants):
        index_by_identity = {id(client): index for index, client in enumerate(self.client_nodes)}
        for client in participants:
            console.info(f"\n[{client.node_id}] Local training started")
            updated_weights, train_record = client.strategy.run_local_training()
            ModelUtils.release_gpu_memory()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record,
                "client_id": str(client.node_id),
                "client_index": int(index_by_identity[id(client)]),
            }

    def run(self) -> None:
        super().run()
        node_vars = self.server_node.node_var
        logger = getattr(node_vars, "training_logger", None)
        weight = getattr(node_vars, "model_weight", None)
        total_rounds = int(self.args.key_value_dict.data["training_rounds"])
        checkpoint_path = None
        if logger is not None and weight is not None:
            checkpoint_path = logger.save_weights_if_enabled(weight, total_rounds)
        if checkpoint_path:
            console.info(f"[RBLA analysis] Final checkpoint saved: {checkpoint_path}")
            finalize = getattr(self.server_node.strategy, "finalize_analysis", None)
            if callable(finalize):
                artifacts = finalize(checkpoint_path)
                console.info(f"[RBLA analysis] Coverage artifacts: {artifacts}")
            metadata = dict(node_vars.config_dict.get("reference_run_metadata", {}))
            metadata["checkpoint_path"] = str(Path(checkpoint_path).resolve())
            metadata["coverage_artifacts"] = artifacts if callable(finalize) else {}
            sidecar = Path(checkpoint_path).with_suffix("")
            metadata_path = Path(f"{sidecar}_run_metadata.json")
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            console.info(f"[RBLA analysis] Run metadata: {metadata_path}")
