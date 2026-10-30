"""Shared HF download/cache handling for the five repaired dataset loaders."""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import numpy as np
import torch


# Pin data revisions so a future Hub update cannot silently change an experiment.
HF_DATASET_REVISIONS = {
    "nyu-mll/glue": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    "fancyzhx/ag_news": "eb185aade064a813bc0b7f42de02595523103ca4",
    "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31",
    "tanganke/kmnist": "6b345bef535b98131f1d945d552906452d19b2f9",
    "flwrlabs/cinic10": "44c23bab233087df95aad862c5b6fa0e5e6030cf",
}


def load_hf_dataset(repo_id: str, root: str, is_download: bool = True, name: str | None = None):
    """Prepare data without changing the caller's training random streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return _load_hf_dataset(repo_id, root, is_download, name)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _load_hf_dataset(repo_id: str, root: str, is_download: bool, name: str | None):
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

    cache_dir = Path(root) / "hf_datasets"
    revision = HF_DATASET_REVISIONS[repo_id]
    if is_download:
        kwargs = {"cache_dir": str(cache_dir), "revision": revision}
        if name is not None:
            kwargs["name"] = name
        return load_dataset(repo_id, **kwargs)

    # Read prepared Arrow files directly; load_dataset can request Hub metadata
    # even when DownloadConfig.local_files_only is set.
    prefixes = [repo_id.replace("/", "___")]
    if repo_id == "stanfordnlp/imdb":
        prefixes.append("imdb")  # The old unqualified ID used this cache name.
    candidates = []
    for prefix in prefixes:
        for info_path in (cache_dir / prefix).glob("*/*/*/dataset_info.json"):
            if name is not None and info_path.parents[2].name != name:
                continue
            info = json.loads(info_path.read_text(encoding="utf-8"))
            source_revisions = {
                match.group(1)
                for url in (info.get("download_checksums") or {})
                if (match := re.search(r"hf://datasets/[^@]+@([0-9a-f]{40})/", url))
            }
            if source_revisions and revision not in source_revisions:
                continue
            candidates.append((info_path.parent, info))
    if not candidates:
        raise FileNotFoundError(f"HF cache for {repo_id} ({revision}) is missing in {cache_dir}. Enable is_download to prepare it.")
    if len(candidates) != 1:
        raise RuntimeError(f"Multiple HF cache versions found for {repo_id}; keep one matching prepared version before offline loading.")

    folder, info = candidates[0]
    dataset_name = info.get("dataset_name") or repo_id.split("/")[-1]
    splits = {}
    for split in ("train", "validation", "test"):
        if split not in info["splits"]:
            continue
        paths = sorted(folder.glob(f"{dataset_name}-{split}*.arrow"))
        if not paths:
            raise FileNotFoundError(f"HF {repo_id}/{split} cache is missing in {folder}")
        shards = [Dataset.from_file(str(path)) for path in paths]
        splits[split] = concatenate_datasets(shards) if len(shards) > 1 else shards[0]
        expected = info["splits"][split]["num_examples"]
        if len(splits[split]) != expected:
            raise RuntimeError(f"Incomplete HF {repo_id}/{split} cache: expected {expected} samples, found {len(splits[split])}.")
    if not {"train", "test"}.issubset(splits):
        raise RuntimeError(f"HF {repo_id} cache must contain train and test splits.")
    return DatasetDict(splits)
