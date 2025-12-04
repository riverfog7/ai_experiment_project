from pathlib import Path
from typing import Optional

from datasets import Image, Value, Features
from pydantic import BaseModel


class ImageClassificationData(BaseModel):
    image_path: Optional[Path]
    class_name: str

    def to_hf_dict(self) -> dict:
        if self.image_path is None:
            raise ValueError("image_path is not set. Please set it before calling to_hf_dict().")
        return {
            "image": self.image_path.absolute().as_posix(),
            "class_name": self.class_name,
        }


HF_FEATURES = Features({
    "image": Image(),
    "class_name": Value("string"),
})
