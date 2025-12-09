from pathlib import Path
from typing import Optional

from datasets import Image, Value, Features
from pydantic import BaseModel


class ImageClassificationData(BaseModel):
    # schema for a single image classification data point
    image_path: Optional[Path]
    class_name: str

    def to_hf_dict(self) -> dict:
        # helper function to convert to a huggingface dataset
        if self.image_path is None:
            raise ValueError("image_path is not set. Please set it before calling to_hf_dict().")
        return {
            "image": self.image_path.absolute().as_posix(),
            "class_name": self.class_name,
        }


# Pass feature types to Dataset
HF_FEATURES = Features({
    "image": Image(),   # This makes huggingface treat this field as a binary image data
    "class_name": Value("string"),
})
