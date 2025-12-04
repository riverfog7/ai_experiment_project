from pathlib import Path
from typing import Optional, Tuple, List

from datasets import Image, Value, Sequence, Features
from pydantic import BaseModel, field_validator, Field


class DetectedObject(BaseModel):
    class_name: str
    bbox: Tuple[float, float, float, float] = Field(..., description="Bounding box in the format (x_min, y_min, x_max, y_max) with values between 0.0 and 1.0")

    @field_validator('bbox')
    @classmethod
    def validate_bbox(cls, v: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        if len(v) != 4:
            raise ValueError(f"Bounding box must contain exactly 4 values (x_min, y_min, x_max, y_max). Got: {v}")
        x_min, y_min, x_max, y_max = v
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid bounding box coordinates: {v}. Ensure that x_min < x_max and y_min < y_max.")
        if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
            raise ValueError(f"Bounding box coordinates must be non-negative. Got: {v}")
        if x_max > 1.0 or y_max > 1.0 or x_min > 1.0 or y_min > 1.0:
            raise ValueError(f"Bounding box coordinates must be within the range [0.0, 1.0]. Got: {v}")
        return v


class ObjectDetectionData(BaseModel):
    image_path: Optional[Path]
    detected_objects: List[DetectedObject]

    def to_hf_dict(self) -> dict:
        if self.image_path is None:
            raise ValueError("image_path is not set. Please set it before calling to_hf_dict().")
        dump = self.model_dump()
        return {
            "image": self.image_path.absolute().as_posix(),
            "detected_objects": dump["detected_objects"],
        }


HF_FEATURES = Features({
    "image": Image(),
    "detected_objects": [
        {
            "class_name": Value("string"),
            "bbox": Sequence(Value("float32"), length=4),
        }
    ],
})
