import json
from pathlib import Path
from typing import Tuple

import numpy as np
from datasets import DatasetDict

from .constants import PRUNE_ITEM_COUNT_THRESHOLD
from .data_augmentations import setup_dataset_transforms


def convert_multitask(class_names: list) -> dict:
    # helper function to convert single class_name to multitask labels
    # drop the last "shape" part of the class_name
    material_labels = ["_".join(name.split("_")[0:2]) for name in class_names]
    transparency_labels = [name.split("_")[2] for name in class_names]

    return {
        "material_class_name": material_labels,
        "transparency_class_name": transparency_labels,
    }

def prune_and_convert_to_multitask(hf_dataset: DatasetDict) -> DatasetDict:
    # helper function to remove classes that exist only on one split or have low item count

    # batched processing with one column for computation speedup
    hf_dataset = hf_dataset.map(convert_multitask, batched=True, batch_size=32768, input_columns=["class_name"])

    # do not prune transparency. Expect two classes only.
    train_transparency_unique = set(hf_dataset['train'].unique("transparency_class_name"))
    val_transparency_unique = set(hf_dataset["validation"].unique("transparency_class_name"))
    assert len(train_transparency_unique) == 2, "Expected 2 unique transparency labels in training set."
    assert len(val_transparency_unique) == 2, "Expected 2 unique transparency labels in validation set."

    to_prune_materials = set()

    # prune samples with materials/transparencies only in one split
    train_material_unique = set(hf_dataset['train'].unique("material_class_name"))
    val_material_unique = set(hf_dataset["validation"].unique("material_class_name"))
    only_train_materials = train_material_unique - val_material_unique
    only_val_materials = val_material_unique - train_material_unique
    only_in_either_split_materials = only_train_materials.union(only_val_materials)
    in_both_splits_materials = train_material_unique.intersection(val_material_unique)
    to_prune_materials.update(only_in_either_split_materials)

    # prune samples with materials having less than threshold items in train or val split
    train_mat_ds = hf_dataset["train"].select_columns(["material_class_name"])
    val_mat_ds = hf_dataset["validation"].select_columns(["material_class_name"])
    train_mat_counts = train_mat_ds.to_pandas()["material_class_name"].value_counts()
    val_mat_counts = val_mat_ds.to_pandas()["material_class_name"].value_counts()
    for material in in_both_splits_materials:
        train_count = train_mat_counts.get(material, 0)
        val_count = val_mat_counts.get(material, 0)
        if train_count < PRUNE_ITEM_COUNT_THRESHOLD or val_count < PRUNE_ITEM_COUNT_THRESHOLD:
            to_prune_materials.add(material)

    print(f"""Pruning the following materials due to low item count or presence in only one split:

{'\n'.join(sorted(list(to_prune_materials)))}
""")

    def filter_fn(mat_class_names: list) -> list:
        keep_mask = [mat not in to_prune_materials for mat in mat_class_names]
        return keep_mask

    hf_dataset["train"] = hf_dataset["train"].filter(filter_fn, batched=True, batch_size=163840, input_columns=["material_class_name"])
    hf_dataset["validation"] = hf_dataset["validation"].filter(filter_fn, batched=True, batch_size=163840, input_columns=["material_class_name"])

    return hf_dataset

def get_class_mappings(hf_dataset: DatasetDict) -> Tuple[dict, dict, dict, dict]:
    # helper function to get label2id, id2label for each classification task
    train_materials = set(hf_dataset["train"].unique("material_class_name"))
    val_materials = set(hf_dataset["validation"].unique("material_class_name"))
    assert train_materials == val_materials, "Train and Validation sets must have the same materials."

    train_transparencies = set(hf_dataset["train"].unique("transparency_class_name"))
    val_transparencies = set(hf_dataset["validation"].unique("transparency_class_name"))
    assert train_transparencies == val_transparencies, "Train and Validation sets must have the same transparencies."

    materials = sorted(list(train_materials))
    transparencies = sorted(list(train_transparencies))

    material_label2id = {str(label): int(idx) for idx, label in enumerate(materials)}
    material_id2label = {int(idx): str(label) for label, idx in material_label2id.items()}
    transparency_label2id = {str(label): int(idx) for idx, label in enumerate(transparencies)}
    transparency_id2label = {int(idx): str(label) for label, idx in transparency_label2id.items()}

    return material_label2id, material_id2label, transparency_label2id, transparency_id2label

def save_class_mappings(save_dir: Path | str, mat_label2id: dict, mat_id2label: dict, trans_label2id: dict, trans_id2label: dict):
    save_dir = Path(save_dir).resolve()
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    label2id = {
        "material_label2id": mat_label2id,
        "transparency_label2id": trans_label2id,
    }
    id2label = {
        "material_id2label": mat_id2label,
        "transparency_id2label": trans_id2label,
    }
    with open(save_dir / "label2id.json", "w") as f:
        json.dump(label2id, f)
    with open(save_dir / "id2label.json", "w") as f:
        json.dump(id2label, f)

def load_class_mappings(load_dir: Path | str) -> Tuple[dict, dict, dict, dict]:
    load_dir = Path(load_dir).resolve()
    with open(load_dir / "label2id.json", "r") as f:
        label2id = json.load(f)
    with open(load_dir / "id2label.json", "r") as f:
        id2label = json.load(f)

    material_label2id = label2id["material_label2id"]
    transparency_label2id = label2id["transparency_label2id"]
    material_id2label = id2label["material_id2label"]
    transparency_id2label = id2label["transparency_id2label"]

    material_id2label = {int(k): str(v) for k, v in material_id2label.items()}
    transparency_id2label = {int(k): str(v) for k, v in transparency_id2label.items()}
    material_label2id = {str(k): int(v) for k, v in material_label2id.items()}
    transparency_label2id = {str(k): int(v) for k, v in transparency_label2id.items()}

    return material_label2id, material_id2label, transparency_label2id, transparency_id2label

def prune_smart_fast(dataset, ratio=0.1, threshold=5000, seed=42):
    """Prune dataset (train set) to preserve all rare classes
    and randomly sample common classes to achieve target ratio."""
    df = dataset.select_columns(["material_class_name"]).to_pandas()
    counts = df["material_class_name"].value_counts()

    is_rare_label = counts < threshold
    needed_common = (len(dataset) * ratio) - counts[is_rare_label].sum()
    rate = max(0.0, needed_common / counts[~is_rare_label].sum())

    # vectorized operations for efficiency
    # is_rare_row is a boolean mask for rows with rare labels
    is_rare_row = df["material_class_name"].isin(counts[is_rare_label].index)
    rng = np.random.default_rng(seed)

    # we preserve all rare rows
    # and sample common rows with probability = rate
    # numpy will generate an array of random numbers between 0 and 1
    # for each row, we keep if its random number < rate
    keep_mask = is_rare_row | (rng.random(len(df)) < rate)

    return dataset.select(df[keep_mask].index).shuffle(seed=seed)

def easy_load(data_path: Path | str,
              img_size: int = 224,
              cache_dir: Path | str = "./datasets/cache",
              include_all_columns: bool = False,
              keep_in_memory: bool = False,
              prune_train_set: bool = False,
              prune_kwargs: dict = {},
) -> Tuple[DatasetDict, dict, dict, dict, dict]:
    # helper function to preprocess dataset and load it
    # utilizes disk caching for faster subsequent loads
    data_path = Path(data_path).resolve()
    cache_dir = Path(cache_dir).resolve()

    id2label = {}
    # if cache exists, load from cache
    # else, process and save to cache
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        hf_dataset = DatasetDict.load_from_disk(data_path)
        hf_dataset = prune_and_convert_to_multitask(hf_dataset)
        hf_dataset.save_to_disk(cache_dir / "easy_load_cache")
        hf_dataset = DatasetDict.load_from_disk(cache_dir / "easy_load_cache", keep_in_memory=keep_in_memory)
        material_label2id, material_id2label, transparency_label2id, transparency_id2label = get_class_mappings(hf_dataset)
        save_class_mappings(
            cache_dir,
            material_label2id,
            material_id2label,
            transparency_label2id,
            transparency_id2label,
        )
        if prune_train_set:
            hf_dataset["train"] = prune_smart_fast(
                hf_dataset["train"],
                **prune_kwargs,
            )
        hf_dataset = setup_dataset_transforms(hf_dataset, material_label2id, transparency_label2id, img_size, include_all_columns)

    else:
        hf_dataset = DatasetDict.load_from_disk(cache_dir / "easy_load_cache", keep_in_memory=keep_in_memory)
        material_label2id, material_id2label, transparency_label2id, transparency_id2label = load_class_mappings(cache_dir)
        if prune_train_set:
            hf_dataset["train"] = prune_smart_fast(
                hf_dataset["train"],
                **prune_kwargs,
            )
        hf_dataset = setup_dataset_transforms(hf_dataset, material_label2id, transparency_label2id, img_size, include_all_columns)

    return hf_dataset, material_label2id, material_id2label, transparency_label2id, transparency_id2label
