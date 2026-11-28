"""Offline HF migration regressions: data contracts, RNG and NonIID partitioning.

Only in-memory or temporary Arrow fixtures; no real caches, downloads or training.
"""
from __future__ import annotations

from collections import Counter
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace
import urllib.request
from unittest.mock import Mock

import datasets as hf
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterDataPipe, Subset
from torch.utils.data.datapipes.iter import IterableWrapper

from flex.fl_algorithms.noniid.noniid_data_generator import NoniidDataGenerator
from flex.ml_data_loader.loader import _dataset_loader_agnews as agnews
from flex.ml_data_loader.loader import _dataset_loader_cinic10 as cinic
from flex.ml_data_loader.loader import _dataset_loader_imdb as imdb
from flex.ml_data_loader.loader import _dataset_loader_kmnist as kmnist
from flex.ml_data_loader.loader import _dataset_loader_mrpc as mrpc
from flex.ml_data_loader.loader import _hf_dataset as hf_cache
from flex.ml_data_loader.loader._hf_image_dataset import HFImageDataset


@pytest.fixture(autouse=True)
def offline_and_restore_rng(monkeypatch):
    import requests

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline regression attempted a download or prompt")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(urllib.request, "urlretrieve", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(hf, "load_dataset", forbidden)
    monkeypatch.setattr("builtins.input", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


class TinyTokenizer:
    vocab_size = 10000
    pad_token_id = 0
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __init__(self):
        self.calls = []

    def __call__(self, texts, pairs=None, **kwargs):
        self.calls.append((texts, pairs, kwargs))
        def token(text):
            number = re.search(r"\d+$", text)
            return int(number[0]) + 1 if number else len(text)
        ids = torch.tensor([[token(text), token(pairs[i]) if pairs else 0]
                            for i, text in enumerate(texts)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                "token_type_ids": torch.tensor([[0, int(pairs is not None)]] * len(texts))}


def text_args(root, **overrides):
    values = dict(root=str(root), is_download=False, batch_size=2,
                  test_batch_size=2, shuffle=False, num_workers=0,
                  train_split="train", test_split="test", max_len=32,
                  tokenizer=TinyTokenizer(), vocab=None)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def mrpc_data():
    def rows(indices, test=False):
        return hf.Dataset.from_list([
            {"idx": i, "label": int(i >= (578 if test else 1323)),
             "sentence1": f"sentence a {i}", "sentence2": f"sentence b {i}"}
            for i in indices
        ])
    validation = set(range(0, 4076, 10))
    return hf.DatasetDict({
        "train": rows([i for i in reversed(range(4076)) if i not in validation]),
        "validation": rows(sorted(validation, reverse=True)),
        "test": rows(reversed(range(1725)), test=True),
    })


@pytest.fixture
def news_data():
    rows = hf.Dataset.from_dict({"label": [0, 1, 2, 3],
                                 "text": ['Title, with comma Body with "quotes"',
                                          "Two multi\nline body", "Three ", "Four description"]})
    return hf.DatasetDict(train=rows, test=rows)


def test_mrpc_reconstructs_complete_original_training_order(mrpc_data, monkeypatch, tmp_path):
    load = Mock(return_value=mrpc_data)
    monkeypatch.setattr(mrpc, "load_hf_dataset", load)
    args = text_args(tmp_path, test_split="validation")
    loader = mrpc.DatasetLoader_MRPC().create(args)
    assert load.call_count == 1
    assert load.call_args.args[0] == "nyu-mll/glue"
    assert load.call_args.kwargs.get("name") == "mrpc"
    assert len(loader._dataset) == 4076
    assert len(loader._test_dataset) == 1725
    assert [row["idx"] for row in loader._dataset] == list(range(4076))
    assert [row["text_a"] for row in loader._dataset] == [f"sentence a {i}" for i in range(4076)]
    assert list(loader._dataset[0]) == ["label", "text_a", "text_b", "idx"]
    assert loader.train_label_distribution == {0: 1323, 1: 2753}
    batch, labels = next(iter(loader.data_loader))
    assert labels.tolist() == [0, 0]
    assert args.tokenizer.calls[0][:2] == (["sentence a 0", "sentence a 1"],
                                          ["sentence b 0", "sentence b 1"])
    assert batch["token_type_ids"].tolist() == [[0, 1], [0, 1]]


@pytest.mark.parametrize("broken", ["duplicate_idx", "missing_row", "unlabeled_test"])
def test_mrpc_rejects_budget_or_idx_changes(mrpc_data, monkeypatch, tmp_path, broken):
    if broken == "duplicate_idx":
        values = mrpc_data["validation"].to_dict()
        values["idx"][0] = values["idx"][1]
        mrpc_data["validation"] = hf.Dataset.from_dict(values)
    elif broken == "missing_row":
        mrpc_data["train"] = mrpc_data["train"].select(range(len(mrpc_data["train"]) - 1))
    else:
        values = mrpc_data["test"].to_dict()
        values["label"][0] = -1
        mrpc_data["test"] = hf.Dataset.from_dict(values)
    monkeypatch.setattr(mrpc, "load_hf_dataset", Mock(return_value=mrpc_data))
    split = "test" if broken == "unlabeled_test" else "train"
    with pytest.raises((ValueError, RuntimeError)):
        list(mrpc._load_mrpc_split(str(tmp_path), split, False))


def test_agnews_retains_text_and_label_contract(news_data, monkeypatch, tmp_path):
    load = Mock(return_value=news_data)
    monkeypatch.setattr(agnews, "load_hf_dataset", load)
    args = text_args(tmp_path)
    loader = agnews.DatasetLoader_Agnews().create(args)
    assert load.call_count == 1
    assert load.call_args.args[0] == "fancyzhx/ag_news"
    assert isinstance(loader._dataset, IterDataPipe)
    expected = [(i + 1, text) for i, text in enumerate(news_data["train"]["text"])]
    assert list(loader._dataset) == expected
    with pytest.raises(TypeError):
        len(loader._dataset)  # Keep the old stream's unknown-length behavior.
    assert loader.data_sample_num is None
    assert loader.train_label_distribution == {1: 1, 2: 1, 3: 1, 4: 1}
    assert [int(label) for _, labels in loader.data_loader for label in labels] == [0, 1, 2, 3]


def identity(batch):
    return batch


@pytest.mark.parametrize("shuffle", [False, True])
def test_agnews_old_datapipe_shuffle_and_model_rng(news_data, monkeypatch, tmp_path, shuffle):
    monkeypatch.setattr(agnews, "load_hf_dataset", Mock(return_value=news_data))
    records = [(int(row["label"]) + 1, row["text"]) for row in news_data["train"]]

    def observe(make_pipe):
        torch.manual_seed(127)
        loader = DataLoader(make_pipe(), batch_size=2, shuffle=shuffle,
                            num_workers=0, collate_fn=identity)
        before = torch.nn.Linear(3, 2).weight.detach().clone()
        epochs = [list(loader), list(loader)]
        after = torch.nn.Linear(3, 2).weight.detach().clone()
        return epochs, before, after, torch.get_rng_state()

    # Cached-data suffix of torchtext 0.18 AG_NEWS: one default-buffer shuffler.
    reference = observe(lambda: IterableWrapper(records).shuffle().set_shuffle(False).sharding_filter())
    actual = observe(lambda: agnews._load_agnews_split(str(tmp_path), "train", False))
    assert actual[0] == reference[0]
    for left, right in zip(actual[1:], reference[1:]):
        assert torch.equal(left, right)


def test_imdb_preserves_rows_and_zero_based_labels(monkeypatch, tmp_path):
    data = hf.DatasetDict({"train": hf.Dataset.from_dict({"text": ["negative", "positive"], "label": [0, 1]}),
                           "test": hf.Dataset.from_dict({"text": ["test positive"], "label": [1]})})
    load = Mock(return_value=data)
    monkeypatch.setattr(imdb, "load_hf_dataset", load)
    loader = imdb.DatasetLoader_Imdb().create(text_args(tmp_path))
    assert load.call_args.args[0] == "stanfordnlp/imdb"
    assert list(loader._dataset) == [("negative", 0), ("positive", 1)]
    assert next(iter(loader.data_loader))[1].tolist() == [0, 1]
    assert next(iter(loader.test_data_loader))[1].tolist() == [1]


def write_arrow_cache(root, repo="stanfordnlp/imdb", name="plain_text", version="fixture"):
    import pyarrow as pa

    builder = repo.split("/")[-1]
    folder = root / "hf_datasets" / repo.replace("/", "___") / name / "1.0.0" / version
    folder.mkdir(parents=True)
    records = {"train": [{"text": "negative", "label": 0}, {"text": "positive", "label": 1}],
               "test": [{"text": "test positive", "label": 1}]}
    for split, rows in records.items():
        table = pa.Table.from_pylist(rows)
        with pa.OSFile(str(folder / f"{builder}-{split}.arrow"), "wb") as handle:
            with pa.ipc.new_stream(handle, table.schema) as writer:
                writer.write_table(table)
    (folder / "dataset_info.json").write_text(json.dumps({"builder_name": builder, "dataset_name": builder,
        "config_name": name, "splits": {split: {"num_examples": len(rows)} for split, rows in records.items()}}),
        encoding="utf-8")
    return folder, records


@pytest.mark.parametrize("repo,name", [
    ("nyu-mll/glue", "mrpc"), ("fancyzhx/ag_news", None), ("stanfordnlp/imdb", None),
    ("tanganke/kmnist", "kmnist"), ("flwrlabs/cinic10", None),
])
def test_shared_online_loader_uses_qualified_repo_revision_and_cache(monkeypatch, tmp_path, repo, name):
    result = object()
    load = Mock(return_value=result)
    monkeypatch.setattr(hf, "load_dataset", load)
    assert hf_cache.load_hf_dataset(repo, str(tmp_path), True, name=name) is result
    assert load.call_args.args[0] == repo
    assert load.call_args.kwargs["cache_dir"] == str(tmp_path / "hf_datasets")
    assert load.call_args.kwargs["revision"] == hf_cache.HF_DATASET_REVISIONS[repo]
    assert load.call_args.kwargs.get("name") == name


@pytest.mark.parametrize("raises", [False, True])
def test_hf_preparation_restores_all_cpu_rng_states_even_on_error(monkeypatch, tmp_path, raises):
    import random
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    def prepare(*args, **kwargs):
        random.random()
        np.random.random(7)
        torch.rand(7)
        if raises:
            raise RuntimeError("fixture preparation failed")
        return "prepared"

    monkeypatch.setattr(hf, "load_dataset", prepare)
    if raises:
        with pytest.raises(RuntimeError, match="fixture preparation failed"):
            hf_cache.load_hf_dataset("stanfordnlp/imdb", str(tmp_path), True)
    else:
        assert hf_cache.load_hf_dataset("stanfordnlp/imdb", str(tmp_path), True) == "prepared"
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("repo,name", [("stanfordnlp/imdb", "plain_text"), ("fancyzhx/ag_news", "default")])
def test_shared_offline_arrow_has_no_network_or_cache_writes(tmp_path, repo, name):
    folder, expected = write_arrow_cache(tmp_path, repo, name)
    before = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    state = torch.get_rng_state().clone()
    actual = hf_cache.load_hf_dataset(repo, str(tmp_path), False)
    assert {split: list(rows) for split, rows in actual.items()} == expected
    assert torch.equal(state, torch.get_rng_state())
    after = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert before == after


def test_shared_offline_rejects_missing_ambiguous_or_incomplete_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        hf_cache.load_hf_dataset("stanfordnlp/imdb", str(tmp_path), False)
    folder, _ = write_arrow_cache(tmp_path)
    metadata = folder / "dataset_info.json"
    info = json.loads(metadata.read_text(encoding="utf-8"))
    info["splits"]["train"]["num_examples"] = 99
    metadata.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(RuntimeError):
        hf_cache.load_hf_dataset("stanfordnlp/imdb", str(tmp_path), False)
    write_arrow_cache(tmp_path, version="other")
    with pytest.raises(RuntimeError):
        hf_cache.load_hf_dataset("stanfordnlp/imdb", str(tmp_path), False)


def test_shared_offline_filters_wrong_revision_and_configuration(tmp_path):
    folder, _ = write_arrow_cache(tmp_path, "nyu-mll/glue", "other")
    with pytest.raises(FileNotFoundError):
        hf_cache.load_hf_dataset("nyu-mll/glue", str(tmp_path), False, name="mrpc")
    folder, _ = write_arrow_cache(tmp_path, "nyu-mll/glue", "mrpc")
    metadata = folder / "dataset_info.json"
    info = json.loads(metadata.read_text(encoding="utf-8"))
    info["download_checksums"] = {"hf://datasets/nyu-mll/glue@" + "0" * 40 + "/file.parquet": {}}
    metadata.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        hf_cache.load_hf_dataset("nyu-mll/glue", str(tmp_path), False, name="mrpc")


def image_split(mode="RGB", labels=(0, 1, 0, 1, 0, 1, 0, 1), paths=None, classes=None):
    images = []
    for index in range(len(labels)):
        image = Image.new(mode, (28, 28) if mode == "L" else (32, 32), index + 1)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        images.append({"bytes": stream.getvalue(), "path": paths[index] if paths else None})
    names = classes or [str(i) for i in range(max(labels) + 1)]
    return hf.Dataset.from_dict({"image": images, "label": list(labels)}, features=hf.Features({
        "image": hf.Image(), "label": hf.ClassLabel(names=names),
    }))


class ChangingTransform:
    def __init__(self):
        self.calls = 0
        self.modes = []

    def __call__(self, image):
        self.calls += 1
        self.modes.append(image.mode)
        pixel = image.getpixel((0, 0))
        return torch.tensor([pixel if isinstance(pixel, int) else pixel[0], self.calls])


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_hf_images_keep_noniid_counts_order_rng_and_lazy_transforms(mode):
    source = image_split(mode)
    transform = ChangingTransform()
    torch.manual_seed(901)
    reference_model = torch.nn.Linear(3, 2).weight.detach().clone()
    torch.manual_seed(901)
    wrapped = HFImageDataset(source, transform=transform, mode=mode)
    loader = DataLoader(wrapped, batch_size=3, shuffle=True,
                        generator=torch.Generator().manual_seed(42), num_workers=0)
    assert torch.equal(reference_model, torch.nn.Linear(3, 2).weight.detach())
    assert wrapped.targets == [0, 1, 0, 1, 0, 1, 0, 1]
    assert transform.calls == 0
    state = torch.get_rng_state().clone()
    generator = NoniidDataGenerator(loader)
    clients = generator.generate_noniid_data(distribution="custom", data_volum_list=[[2, 1], [2, 3]],
                                              verify_allocate=False)
    assert torch.equal(state, torch.get_rng_state())
    assert transform.calls == 0
    assert all(isinstance(client, Subset) and client.dataset is wrapped for client in clients)
    indices = [index for client in clients for index in client.indices]
    assert sorted(indices) == list(range(8))
    assert [Counter(wrapped.targets[i] for i in client.indices) for client in clients] == [
        {0: 2, 1: 1}, {0: 2, 1: 3}]
    reference_order = [i for batch in DataLoader(range(8), batch_size=3, shuffle=True,
                                                generator=torch.Generator().manual_seed(42)).batch_sampler for i in batch]
    assert generator.x_train.tolist() == reference_order
    first = clients[0][0][0]
    second = clients[0][0][0]
    assert first[0] == second[0] and first[1] != second[1]
    assert transform.modes == [mode, mode]
    assert wrapped.source_order == "hf"
    assert wrapped.historical_equivalence_verified is False


@pytest.mark.parametrize("module,loader_class,repo,mode", [
    (kmnist, "DatasetLoader_Kmnist", "tanganke/kmnist", "L"),
    (cinic, "DatasetLoader_Cinic10", "flwrlabs/cinic10", "RGB"),
])
def test_image_loaders_use_hf_splits_and_keep_transforms(module, loader_class, repo, mode, monkeypatch, tmp_path):
    classes = cinic._CINIC10_CLASSES if module is cinic else ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]
    split = image_split(mode, labels=tuple(range(10)), classes=classes)
    data = hf.DatasetDict(train=split, test=split, validation=split)
    load = Mock(return_value=data)
    monkeypatch.setattr(module, "load_hf_dataset", load)
    train_transform, test_transform = ChangingTransform(), ChangingTransform()
    args = SimpleNamespace(root=str(tmp_path), is_download=False, is_train=True, split="",
                           batch_size=2, test_batch_size=3, shuffle=False, num_workers=0,
                           transform=train_transform, test_transform=test_transform, generator_seed=42)
    loader = getattr(module, loader_class)().create(args)
    assert all(call.args[0] == repo for call in load.call_args_list)
    assert load.call_count == 1
    assert loader.data_loader.batch_size == 2
    assert loader.test_data_loader.batch_size == 3
    assert isinstance(loader._dataset, HFImageDataset)
    assert loader._dataset.transform is train_transform
    assert loader._test_dataset.transform is test_transform
    assert train_transform.calls == test_transform.calls == 0
    assert next(iter(loader.data_loader))[1].tolist() == [0, 1]
    assert train_transform.modes == [mode, mode]
    assert loader._dataset.class_to_idx == dict(zip(classes, range(10)))


@pytest.mark.parametrize("complete_paths", [False, True])
def test_cinic_path_order_is_conditional_and_never_claims_historical_equivalence(complete_paths):
    paths = ["train/cat/b.png", "train/airplane/b.png", "train/airplane/a.png"] if complete_paths else None
    split = image_split(labels=(3, 0, 0), paths=paths, classes=cinic._CINIC10_CLASSES)
    wrapped = cinic._cinic10_image_dataset(split, None, "train")
    assert wrapped.targets == ([0, 0, 3] if complete_paths else [3, 0, 0])
    assert [wrapped[i][0].getpixel((0, 0))[0] for i in range(3)] == ([3, 2, 1] if complete_paths else [1, 2, 3])
    assert wrapped.source_order == ("imagefolder_paths" if complete_paths else "hf")
    assert wrapped.historical_equivalence_verified is False


def test_cinic_valid_alias_uses_validation_not_training(monkeypatch, tmp_path):
    train = image_split(labels=(0,), classes=cinic._CINIC10_CLASSES)
    valid = image_split(labels=(1,), classes=cinic._CINIC10_CLASSES)
    test = image_split(labels=(2,), classes=cinic._CINIC10_CLASSES)
    monkeypatch.setattr(cinic, "load_hf_dataset", Mock(return_value=hf.DatasetDict(train=train, validation=valid, test=test)))
    args = SimpleNamespace(root=str(tmp_path), is_download=False, is_train=True, split="valid",
                           batch_size=2, shuffle=False, num_workers=0, transform=ChangingTransform(), generator_seed=42)
    loader = cinic.DatasetLoader_Cinic10().create(args)
    assert loader._dataset.targets == [1]
    assert loader._test_dataset.targets == [2]


def test_nlp_hf_records_keep_pair_fields_through_noniid(mrpc_data, monkeypatch, tmp_path):
    monkeypatch.setattr(mrpc, "load_hf_dataset", Mock(return_value=mrpc_data))
    loader = mrpc.DatasetLoader_MRPC().create(text_args(tmp_path, batch_size=512))
    generator = NoniidDataGenerator(loader.data_loader)
    clients = generator.generate_noniid_data(distribution="custom", data_volum_list=[[1000, 2000], [323, 753]],
                                              verify_allocate=False)
    assert [len(client) for client in clients] == [3000, 1076]
    labels = Counter()
    tokens = []
    for client in clients:
        for inputs, label in client:
            labels[int(label)] += 1
            tokens.append(int(inputs["input_ids"][0]))
            assert inputs["token_type_ids"].tolist() == [0, 1]
            assert inputs["attention_mask"].tolist() == [1, 1]
    assert labels == {0: 1323, 1: 2753}
    assert sorted(tokens) == list(range(1, 4077))


@pytest.mark.parametrize("label_shortage", [False, True])
def test_preflight_checks_existing_noniid_budget(mrpc_data, monkeypatch, tmp_path, label_shortage):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "test" / "check_dataset_preflight.py"
    spec = importlib.util.spec_from_file_location("dataset_preflight_fixture", path)
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    if label_shortage:
        values = mrpc_data["train"].to_dict()
        values["label"][0] = 0  # One fewer positive example than the current matrix requests.
        mrpc_data["train"] = hf.Dataset.from_dict(values)
    monkeypatch.setattr(hf_cache, "load_hf_dataset", Mock(return_value=mrpc_data))
    report = preflight.run_preflight(tmp_path, names=("mrpc",))
    result = report["datasets"]["mrpc"]
    assert result["status"] == ("validation_failed" if label_shortage else "ok")
    assert result["splits"]["train"]["noniid_budget"]["compatible_without_capping"] is (not label_shortage)
    assert result["torch_cpu_rng_unchanged"] is True
    assert len(result["splits"]["train"]["ordered_records_sha256"]) == 64


def test_preflight_full_decode_checks_images_beyond_the_first_batch():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "test" / "check_dataset_preflight.py"
    spec = importlib.util.spec_from_file_location("dataset_preflight_images", path)
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    source = image_split(labels=tuple(range(10)))
    wrapped = HFImageDataset(source)
    sampled = preflight.summarize_images(wrapped, 10, 2, [32, 32, 3])
    complete = preflight.summarize_images(wrapped, 10, 2, [32, 32, 3], full_decode=True)
    assert sampled["decoded_image_count"] == 2
    assert sampled["all_images_decoded"] is False
    assert complete["decoded_image_count"] == 10
    assert complete["all_images_decoded"] is True
    assert complete["first_raw_batch"] == sampled["first_raw_batch"]

    rows = source.cast_column("image", hf.Image(decode=False)).to_dict()
    stream = io.BytesIO()
    Image.new("RGB", (16, 16)).save(stream, format="PNG")
    rows["image"][-1] = {"bytes": stream.getvalue(), "path": None}
    broken = HFImageDataset(hf.Dataset.from_dict(rows, features=source.features))
    assert preflight.summarize_images(broken, 10, 2, [32, 32, 3])["decoded_image_count"] == 2
    with pytest.raises(ValueError, match="Image 9 has shape"):
        preflight.summarize_images(broken, 10, 2, [32, 32, 3], full_decode=True)
