import warnings
from pathlib import Path
from typing import List, Generator, Tuple

from .image_utils import BatchCropper
from .models import ImageDescription, ImageClassificationData, ObjectDetectionData, DetectedObject
from .utils import get_class_name


def convert_to_image_classification(
        image_descriptions: List[ImageDescription],
        cache_dir: Path | str = "./.temp/image_classification",
        size_threshold: Tuple[int, int] = (50, 50),
        pad: int = 5,
) -> Generator[ImageClassificationData, None, None]:
    # generator that converts original data format to image classification format
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cropper = BatchCropper(output_dir=cache_dir, size_threshold=size_threshold, pad=pad)
    results = cropper.process(image_descriptions)

    for img_desc, cropped_paths in zip(image_descriptions, results):
        for idx, cropped_path in enumerate(cropped_paths):
            if cropped_path is None:
                continue
            obj = img_desc.annotation_info[idx]
            yield ImageClassificationData(
                image_path=cropped_path,
                class_name=get_class_name(obj)
            )


def convert_to_object_detection(
        image_descriptions: List[ImageDescription],
        size_threshold: Tuple[int, int] = (50/768, 50/768),
) -> Generator[ObjectDetectionData, None, None]:
    # generator that converts original data format to object detection format
    for img_desc in image_descriptions:
        if img_desc.image_path is None:
            continue

        detected_objects = []
        img_width, img_height = img_desc.image_info.image_width, img_desc.image_info.image_height
        for annotation in img_desc.annotation_info:
            x_min, y_min, x_max, y_max = annotation.get_proper_bbox()
            x_min /= img_width
            y_min /= img_height
            x_max /= img_width
            y_max /= img_height

            x_min = max(0.0, min(1.0, x_min))
            y_min = max(0.0, min(1.0, y_min))
            x_max = max(0.0, min(1.0, x_max))
            y_max = max(0.0, min(1.0, y_max))

            if (x_max - x_min) < size_threshold[0] and (y_max - y_min) < size_threshold[1]:
                warnings.warn(f"Skipping small object with size ({x_max - x_min}, {y_max - y_min})")
                continue

            bbox = (x_min, y_min, x_max, y_max)

            detected_objects.append(
                DetectedObject(
                    class_name=get_class_name(annotation),
                    bbox=bbox
                )
            )

        yield ObjectDetectionData(
            image_path=img_desc.image_path,
            detected_objects=detected_objects
        )
