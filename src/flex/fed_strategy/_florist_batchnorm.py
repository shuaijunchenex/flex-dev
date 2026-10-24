"""Explicit CNN BatchNorm policies; independent of FLoRIST's SVD algorithm."""

from torch import nn


def batchnorm_mode(args):
    mode = args.get("batchnorm_mode", "frozen")
    if mode not in ("frozen", "fedavg"):
        raise ValueError("FLoRIST batchnorm_mode must be frozen or fedavg")
    return mode


def batchnorm_keys(model, mode, head_keys):
    keys, counters = set(), set()
    for prefix, module in model.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if not module.track_running_stats:
            raise ValueError("FLoRIST BatchNorm requires track_running_stats=True")
        if mode == "fedavg" and module.momentum is None:
            raise ValueError("FLoRIST BatchNorm FedAvg requires fixed momentum")
        local = {f"{prefix}.{key}" if prefix else key for key in module.state_dict()}
        if local & head_keys:
            raise ValueError("FLoRIST trainable head overlaps BatchNorm; use batchnorm_mode")
        if mode == "fedavg":
            keys.update(local)
            counters.add(f"{prefix}.num_batches_tracked" if prefix else "num_batches_tracked")
    return keys, counters


def validate_batchnorm_state(payload, keys):
    for key in keys:
        if key.endswith("running_var") and (payload[key] < 0).any():
            raise ValueError(f"FLoRIST BatchNorm running variance must be nonnegative: {key}")


def freeze_batchnorm_forwards(model):
    """Attach to the actual training copy, after deepcopy and before trainer.train."""
    modules = tuple(m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    if not modules:
        return None

    def before_forward(_model, _inputs):
        # The standard trainer calls model.train() every epoch. Keep Dropout active.
        for module in modules:
            module.eval()

    return model.register_forward_pre_hook(before_forward)
