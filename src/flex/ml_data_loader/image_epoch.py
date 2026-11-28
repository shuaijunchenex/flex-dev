"""Repeatable minibatches for a local epoch on a map-style image dataset."""
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler


@contextmanager
def _seeded_cpu_rng(seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.default_generator.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def iter_image_epoch(loader, *, seed_parts, with_indices=False):
    """Yield collated batches, optionally paired with their indices in loader.dataset."""
    if isinstance(loader.dataset, IterableDataset) or loader.batch_sampler is None:
        raise ValueError("A seeded image epoch requires a batched, map-style dataset")

    def seed(*suffix):
        digest = hashlib.sha256(":".join(map(str, (*seed_parts, *suffix))).encode()).digest()
        return int.from_bytes(digest[:8], "little")

    sampler = loader.batch_sampler
    order_seed = seed("order")
    if type(sampler) is BatchSampler and type(sampler.sampler) is RandomSampler:
        source = sampler.sampler
        order = RandomSampler(source.data_source, replacement=source.replacement,
                              num_samples=source.num_samples,
                              generator=torch.Generator().manual_seed(order_seed))
        sampler = BatchSampler(order, sampler.batch_size, sampler.drop_last)
    # Materialize once so custom samplers, repeats and tail batches match the reported IDs.
    with _seeded_cpu_rng(order_seed):
        indices = [list(batch) for batch in sampler]
    epoch_loader = DataLoader(loader.dataset, batch_sampler=indices, collate_fn=loader.collate_fn,
                              num_workers=loader.num_workers, worker_init_fn=loader.worker_init_fn,
                              multiprocessing_context=loader.multiprocessing_context,
                              prefetch_factor=loader.prefetch_factor, timeout=loader.timeout,
                              pin_memory=loader.pin_memory, pin_memory_device=loader.pin_memory_device,
                              generator=torch.Generator().manual_seed(seed("workers")))
    batches = iter(epoch_loader)
    for batch_index, batch_indices in enumerate(indices):
        if loader.num_workers:
            batch = next(batches)
        else:
            with _seeded_cpu_rng(seed(batch_index, "augment")):
                batch = next(batches)
        yield (batch_indices, batch) if with_indices else batch
