from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal, Tuple

from datasets import Features, Value, Sequence, Image
from pydantic import BaseModel, Field, field_validator


class ImageInfo(BaseModel):
    file_name: str = Field(..., alias="FILE_NAME")
    date: datetime = Field(..., alias="DATE")
    resolution: str = Field(..., alias="RESOLUTION")
    image_photographer: str = Field(..., alias="IMAGE_PHOTOGRAPHER")
    focus_distance: Optional[float] = Field(..., alias="FOCUS_DISTANCE")
    exposure_time: Optional[str] = Field(..., alias="EXPOSURE_TIME")
    sensitivity_iso: Optional[int] = Field(..., alias="SENSITIVITY_ISO")
    place: str = Field(..., alias="PLACE")
    region_name: Optional[str] = Field(..., alias="REGION_NAME")
    direction: str = Field(None, alias="Direction")
    height: Optional[int] = Field(..., alias="HEIGHT")
    day_or_night: str = Field(..., alias="DAY/NIGHT")
    weather: str = Field(..., alias="WEATHER")
    image_height: int = Field(..., alias="IMAGE_HEIGHT")
    image_width: int = Field(..., alias="IMAGE_WIDTH")
    image_size: str = Field(..., alias="IMAGE_SIZE")


class AnnotationEntry(BaseModel):
    id: str = Field(..., alias="ID")
    class_name: str = Field(..., alias="CLASS")
    details: str = Field(..., alias="DETAILS")
    damage: str = Field(..., alias="DAMAGE")
    dirtiness: str = Field(..., alias="DIRTINESS")
    cover: str = Field(..., alias="COVER")
    transparency: str = Field(..., alias="TRANSPARENCY")
    shape: str = Field(..., alias="SHAPE")
    shape_type: Literal["BOX", "POLYGON"] = Field(..., alias="SHAPE_TYPE")
    points: List[List[float]] = Field(..., alias="POINTS", min_length=1)

    @field_validator('points')
    @classmethod
    def check_inner_list_length(cls, v: List[List[float]]) -> List[List[float]]:
        # Case: polygon
        if len(v) != 1:
            if len(v) < 4:
                raise ValueError(f"For polygon, points must contain at least 4 points. Got: {len(v)} points.")
            for i, point in enumerate(v):
                if len(point) != 2:
                    raise ValueError(f"Each point must have exactly 2 coordinates (x, y). Error at index {i}: {point}")

        # Case: bounding box
        else:
            if len(v[0]) != 4:
                raise ValueError(f"For bounding box, points must contain exactly 4 values (x_min, y_min, x_max, y_max). Got: {v[0]}")
        return v

    def get_proper_bbox(self) -> Tuple[float, float, float, float]:
        # returns coordinates in pixels: (x_min, y_min, x_max, y_max)
        if self.shape_type == "BOX":
            box = self.points[0]
            return box[0], box[1], box[0] + box[2], box[1] + box[3]
        elif self.shape_type == "POLYGON":
            xs = [point[0] for point in self.points]
            ys = [point[1] for point in self.points]
            return min(xs), min(ys), max(xs), max(ys)
        else:
            raise ValueError(f"Unsupported shape type: {self.shape_type}")


class ImageDescription(BaseModel):
    image_path: Optional[Path] = None # Will be filled later by a helper function
    image_info: ImageInfo = Field(..., alias="IMAGE_INFO")
    annotation_info: List[AnnotationEntry] = Field(..., alias="ANNOTATION_INFO")

    def to_hf_dict(self) -> dict:
        if self.image_path is None:
            raise ValueError("image_path is not set. Please set it before calling to_hf_dict().")
        dump = self.model_dump(by_alias=False)

        return {
            "image": self.image_path.absolute().as_posix(),
            "image_info": dump["image_info"],
            "annotation_info": dump["annotation_info"],
        }


HF_FEATURES = Features({
    "image": Image(),  # The physical image file

    "image_info": {
        "file_name": Value("string"),
        "date": Value("timestamp[us]"),  # HF handles datetime objects automatically
        "resolution": Value("string"),
        "image_photographer": Value("string"),
        "focus_distance": Value("float32"),
        "exposure_time": Value("string"),
        "sensitivity_iso": Value("int32"),
        "place": Value("string"),
        "region_name": Value("string"),
        "direction": Value("string"),
        "height": Value("int32"),
        "day_or_night": Value("string"),
        "weather": Value("string"),
        "image_height": Value("int32"),
        "image_width": Value("int32"),
        "image_size": Value("string"),
    },

    "annotation_info": Sequence({
        "id": Value("string"),
        "class_name": Value("string"),
        "details": Value("string"),
        "damage": Value("string"),
        "dirtiness": Value("string"),
        "cover": Value("string"),
        "transparency": Value("string"),
        "shape": Value("string"),
        "shape_type": Value("string"),
        "points": Sequence(Sequence(Value("float32"))),
    })
})
