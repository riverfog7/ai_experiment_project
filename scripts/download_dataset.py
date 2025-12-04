#!/usr/bin/env -S uv run

import argparse
from pathlib import Path
from typing import List

import dotenv
from tqdm import tqdm

from aie_project.aihub_helper import AIHubHelper
from aie_project.aihub_helper.image_utils import resize_images_parallel

dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())


def download(dataset_id: int, output_dir: Path, log_file: Path):
    helper = AIHubHelper(max_background_workers=4)

    print("Fetching dataset information...")
    dataset = helper.list_dataset(dataset_id)

    # train and validation datasets
    train_ds = dataset[0][0]
    train_images = train_ds[0]
    train_image_keys = [e.file_sn for e in train_images]
    train_labels = train_ds[1]
    train_label_keys = [e.file_sn for e in train_labels]

    val_ds = dataset[0][1]
    val_images = val_ds[0]
    val_image_keys = [e.file_sn for e in val_images]
    val_labels = val_ds[1]
    val_label_keys = [e.file_sn for e in val_labels]

    helper.download_and_extract_file(dataset_key=dataset_id, file_sn=train_label_keys, output_dir=output_dir,
                                     unzip=False)
    helper.download_and_extract_file(dataset_key=dataset_id, file_sn=val_label_keys, output_dir=output_dir, unzip=False)

    completed: List[str] = []
    if log_file.exists():
        with open(log_file, "r") as f:
            completed = [line.strip() for line in f.readlines()]

    try:
        for sn in tqdm(train_image_keys + val_image_keys, desc="Downloading and processing images"):
            if str(sn) in completed:
                tqdm.write(f"Skipping already downloaded file_sn={sn}")
                continue

            helper.download_and_extract_file(
                dataset_key=dataset_id,
                file_sn=sn,
                output_dir=output_dir,
                transform=resize_images_parallel,
                background_processing=True
            )

            completed.append(str(sn))
            tqdm.write(f"Download complete (processing in background) file_sn={sn}")

            with open(log_file, "w") as f:
                for line in completed:
                    f.write(f"{line}\n")
    finally:
        print("\nAll downloads finished. Waiting for background processing to complete...")
        helper.shutdown()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process AIHub dataset images.")
    parser.add_argument("--dataset-id", type=int, default=71362, help="AIHub dataset ID to download.")
    parser.add_argument("--output-dir", type=str, help="Directory to save downloaded data.", required=True)
    parser.add_argument("--log-file", type=str, default="./download_progress.txt",
                        help="File to log download progress.")
    args = parser.parse_args()

    download(dataset_id=args.dataset_id, output_dir=Path(args.output_dir), log_file=Path(args.log_file))
