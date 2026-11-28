from __future__ import annotations
from typing import Any


def make_client_update(node_id: str, updated_weights: dict, train_record: dict, data_sample_num: int, **extra: Any) -> dict:
    """
    Build a client update that every aggregator accepts.

    FedAvg-style aggregators read 'train_record["data_sample_num"]', RBLA-SASG reads the
    top-level 'data_sample_num', so both are filled. 'extra' keys are added as they are.
    """
    return {
        "node_id": node_id,
        "updated_weights": updated_weights,
        "data_sample_num": data_sample_num,
        "train_record": {**train_record, "data_sample_num": data_sample_num},
        **extra,
    }
