import warnings
from pathlib import Path
from typing import Generator, Optional, Literal

from datasets import Dataset, DatasetDict
from pydantic import ValidationError

from .conversion import (
    convert_to_image_classification,
    convert_to_object_detection
)
from .models import ImageDescription, HF_FEATURES
from .models import OD_HF_FEATURES, IC_HF_FEATURES
from .utils import CachedImageFinder


def data_generator(data_subset_path: Path) -> Generator[ImageDescription, None, None]:
    # Load data from JSON files
    data_subset_path = Path(data_subset_path)
    if not data_subset_path.exists():
        raise FileNotFoundError(f"Data subset path not found: {data_subset_path}")

    finder = CachedImageFinder(data_subset_path, ext=".jpg")

    for json_file in data_subset_path.rglob("**/*.json"):
        try:
            desc = ImageDescription.model_validate_json(json_file.read_text())
            img_path = finder.find(desc.image_info.file_name)
            if img_path:
                desc.image_path = img_path
                yield desc
            else:
                warnings.warn(f"Image file not found for {desc.image_info.file_name} referenced in {json_file}")
        except ValidationError as ve:
            warnings.warn(f"Validation error for file {json_file}: {ve}")

    del finder


def hf_data_generator(
        data_subset_path: Path | str,
        convert_to: Optional[Literal["object_detection", "image_classification"]] = None,
        cache_dir: Path | str = "./.temp",
) -> Generator[dict, None, None]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data and convert to Hugging Face format
    data_subset_path = Path(data_subset_path)
    if not data_subset_path.exists():
        raise FileNotFoundError(f"Data subset path not found: {data_subset_path}")

    if convert_to is None:
        for img_desc in data_generator(data_subset_path):
            yield img_desc.to_hf_dict()
    else:
        # we preload all data in memory
        # convert functions use multiprocessing internally
        # which will load all into memory anyway
        img_descs = list(data_generator(data_subset_path))
        if convert_to == "object_detection":
            for od_data in convert_to_object_detection(img_descs):
                yield od_data.to_hf_dict()
        elif convert_to == "image_classification":
            for ic_data in convert_to_image_classification(img_descs, cache_dir=cache_dir):
                yield ic_data.to_hf_dict()
        else:
            raise ValueError(f"Invalid convert_to value: {convert_to}. Must be one of None, 'object_detection', 'image_classification'.")

        # clean up memory
        del img_descs


def subset_to_hf_dataset(
        data_subset_path: Path,
        cache_dir: Path | str = "./.temp",
        convert_to: Optional[Literal["object_detection", "image_classification"]] = "image_classification",
) -> Dataset:
    # Load all data and convert to Hugging Face format
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    if convert_to not in (None, "object_detection", "image_classification"):
        raise ValueError(f"Invalid convert_to value: {convert_to}. Must be one of None, 'object_detection', 'image_classification'.")

    # different features to pass to Dataset based on conversion type
    if convert_to == "object_detection":
        features = OD_HF_FEATURES
    elif convert_to == "image_classification":
        features = IC_HF_FEATURES
    else:
        features = HF_FEATURES

    return Dataset.from_generator(
        generator=hf_data_generator,
        gen_kwargs={
            "data_subset_path": data_subset_path,
            "convert_to": convert_to,
            "cache_dir": cache_dir / "image_cache"
        },
        cache_dir=cache_dir.absolute().as_posix(),
        features=features,
    )


def to_hf_dataset(
        data_root: Path,
        cache_dir: Path | str = "./.temp",
        convert_to: Optional[Literal["object_detection", "image_classification"]] = "image_classification",
):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root path not found: {data_root}")

    # hope we don't have files named Training or Validation
    # Searching the entire filetree will take forever with 2M+ files
    train_root = next((p for p in data_root.rglob("Training") if p.is_dir()), None)
    val_root = next((p for p in data_root.rglob("Validation") if p.is_dir()), None)

    if train_root is None or val_root is None:
        raise ValueError(f"Could not find 'Training' or 'Validation' directories under {data_root}")

    print(f"Found training data at: {train_root}")
    print(f"Found validation data at: {val_root}")

    train_dataset = subset_to_hf_dataset(
        data_subset_path=train_root,
        cache_dir=cache_dir,
        convert_to=convert_to,
    )
    val_dataset = subset_to_hf_dataset(
        data_subset_path=val_root,
        cache_dir=cache_dir,
        convert_to=convert_to,
    )

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })
