#!/usr/bin/env -S uv run

import argparse
import shutil
from pathlib import Path

from aie_project.dataset_helper import to_hf_dataset


def convert(
        dataset_path: Path,
        output_path: Path,
        convert_to: str | None = "image_classification",
        cache_dir: Path | str = "./.temp",
        remove_cache: bool = True,
):
    cache_exists = Path(cache_dir).exists()
    if not cache_exists:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print("Creating Hugging Face dataset...")
    dataset = to_hf_dataset(
        data_root=dataset_path,
        convert_to=convert_to,
        cache_dir=cache_dir,
    )
    print(f"Saving dataset to {output_path} ...")
    dataset.save_to_disk(output_path)

    if remove_cache and not cache_exists:
        print("Removing cache directory...")
        # only remove cache if we created it
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset subset to Hugging Face dataset format.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset subset.")
    parser.add_argument("output_path", type=Path, help="Path to save the Hugging Face dataset.")
    parser.add_argument("--convert_to", type=str, choices=["object_detection", "image_classification", "none"],
                        default="image_classification", help="Type of conversion to perform.")
    parser.add_argument("--cache_dir", type=Path, default=Path("./.temp"), help="Cache directory for temporary files.")
    parser.add_argument("--no_remove_cache", action="store_true", help="Do not remove cache directory after conversion.")

    args = parser.parse_args()

    convert(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        convert_to=None if args.convert_to == "none" else args.convert_to,
        cache_dir=args.cache_dir,
        remove_cache=not args.no_remove_cache,
    )
