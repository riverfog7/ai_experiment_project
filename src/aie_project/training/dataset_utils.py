from pathlib import Path
from typing import Tuple
import json

from datasets import DatasetDict

from .data_augmentations import setup_dataset_transforms


def simplify_class_names(batch) -> dict:
    new_names = [
        "_".join(name.split("_")[:-1])
        for name in batch["class_name"]
    ]
    return {"class_name": new_names}

def prune_and_simplify_dataset(hf_dataset: DatasetDict) -> DatasetDict:
    hf_dataset = hf_dataset.map(simplify_class_names, batched=True)

    train_label_unique = set(hf_dataset["train"].unique("class_name"))
    val_label_unique = set(hf_dataset["validation"].unique("class_name"))
    only_train_labels = train_label_unique - val_label_unique

    def filter_fn(example):
        return example["class_name"] not in only_train_labels

    hf_dataset["train"] = hf_dataset["train"].filter(filter_fn)

    return hf_dataset

def get_class_mappings(hf_dataset: DatasetDict) -> Tuple[dict, dict]:
    train_labels = set(hf_dataset["train"].unique("class_name"))
    val_labels = set(hf_dataset["validation"].unique("class_name"))
    assert train_labels == val_labels, "Train and Validation sets must have the same labels."

    unique_labels = sorted(list(train_labels))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    return label2id, id2label

def easy_load(data_path: Path | str, img_size: int = 224, cache_dir: Path | str = "./datasets/cache") -> Tuple[DatasetDict, dict, dict]:
    data_path = Path(data_path).resolve()
    cache_dir = Path(cache_dir).resolve()

    id2label = {}
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        hf_dataset = DatasetDict.load_from_disk(data_path)
        hf_dataset = prune_and_simplify_dataset(hf_dataset)
        hf_dataset.save_to_disk(cache_dir / "easy_load_cache")
        label2id, id2label = get_class_mappings(hf_dataset)
        with open(cache_dir / "label2id.json", "w") as f:
            json.dump(label2id, f)
        with open(cache_dir / "id2label.json", "w") as f:
            json.dump(id2label, f)
        hf_dataset = setup_dataset_transforms(hf_dataset, label2id, img_size)

    else:
        hf_dataset = DatasetDict.load_from_disk(cache_dir / "easy_load_cache")
        with open(cache_dir / "label2id.json", "r") as f:
            label2id = json.load(f)
        with open(cache_dir / "id2label.json", "r") as f:
            id2label = json.load(f)
        hf_dataset = setup_dataset_transforms(hf_dataset, label2id, img_size)

    return hf_dataset, label2id, id2label
