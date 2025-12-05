from typing import Tuple

from datasets import DatasetDict
from torchvision import transforms
from transformers import ConvNextImageProcessor

from .constants import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def get_train_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # Geometric Augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # Photometric Augmentations (Helps with lighting variance)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        # Final Conversion
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def setup_dataset_transforms(hf_dataset, label2id: dict, img_size=IMG_SIZE, include_all_columns: bool = False) -> DatasetDict:
    train_tfm = get_train_transform(img_size)
    val_tfm = get_val_transform(img_size)

    def preprocess_train(examples):
        examples["pixel_values"] = [train_tfm(image.convert("RGB")) for image in examples["image"]]
        examples["labels"] = [label2id[c] for c in examples["class_name"]]
        if not include_all_columns:
            for key in list(examples.keys()):
                if key not in ["pixel_values", "labels"]:
                    del examples[key]
        return examples

    def preprocess_val(examples):
        examples["pixel_values"] = [val_tfm(image.convert("RGB")) for image in examples["image"]]
        examples["labels"] = [label2id[c] for c in examples["class_name"]]
        if not include_all_columns:
            for key in list(examples.keys()):
                if key not in ["pixel_values", "labels"]:
                    del examples[key]
        return examples

    if "train" in hf_dataset:
        hf_dataset["train"].set_transform(preprocess_train)

    if "validation" in hf_dataset:
        hf_dataset["validation"].set_transform(preprocess_val)

    return hf_dataset


def save_deployment_processor(save_path):
    # TODO: test if this works
    # We use ConvNextImageProcessor because it is a standard, robust generic processor
    # that supports the "Resize -> Rescale -> Normalize" workflow perfectly.
    processor = ConvNextImageProcessor(
        # Resize Logic
        do_resize=True,
        size={"shortest_edge": IMG_SIZE},
        crop_size={"height": IMG_SIZE, "width": IMG_SIZE},
        do_center_crop=True,  # Optional: ensures square aspect ratio

        # Rescale Logic (0-255 -> 0-1)
        do_rescale=True,
        rescale_factor=1 / 255,

        # Normalize Logic
        do_normalize=True,
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
    )

    # Save to the same folder as your model weights
    processor.save_pretrained(save_path)
    print(f"Saved preprocessor_config.json to {save_path}")
