from torchvision import transforms
from transformers import ConvNextImageProcessor

from .configs import EfficientClassificationConfig
from .constants import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transform(config: EfficientClassificationConfig):
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        # Geometric Augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),

        # Photometric Augmentations (Helps with lighting variance)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

        # Final Conversion
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transform(config: EfficientClassificationConfig):
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


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
