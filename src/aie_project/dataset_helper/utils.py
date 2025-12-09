from pathlib import Path
from typing import Tuple

from .models import AnnotationEntry, ImageDescription


def get_scale(img_desc: ImageDescription, img_size: Tuple[int, int]) -> float:
    # get scale of current image compared to original size (for bbox scaling)
    orig_width = img_desc.image_info.image_width
    orig_height = img_desc.image_info.image_height
    img_width, img_height = img_size

    scale_x = img_width / orig_width
    scale_y = img_height / orig_height

    # account for floating point errors
    assert abs(scale_x - scale_y) < 3e-3, f"Non-uniform scaling detected between original and actual image sizes: {scale_x} vs {scale_y}"

    return (scale_x + scale_y) / 2.0


def get_class_name(annotation: AnnotationEntry) -> str:
    # TODO: Change this to final form
    return f"{annotation.class_name}_{annotation.details}_{annotation.transparency}_{annotation.shape}"


class CachedImageFinder:
    def __init__(self, base_path: Path, ext: str = None):
        # fast path lookup for millions of image files
        # Caches the entire file structure in memory for fast lookup
        # and check filesystem only if not found in cache
        self.base_path = base_path
        self.ext = ext
        self.cache = self._build_cache()

    def _build_cache(self):
        # initialize cache by walking through every file
        cache = {}
        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > 0:
                if self.ext is None or file_path.suffix.lower() == self.ext.lower():
                    if file_path.name in cache:
                        # we expect unique file names (otherwise the dataset is corrupted)
                        raise ValueError(f"Duplicate file name found: {file_path.name}")
                    cache[file_path.name] = file_path
        return cache

    def find(self, file_name: str) -> Path | None:
        if file_name in self.cache:
            return self.cache[file_name]

        # if somehow not in cache, do a search
        found = self.base_path.rglob(file_name)
        for path in found:
            if path.is_file() and path.stat().st_size > 0:
                self.cache[file_name] = path
                return path
        return None

    def __del__(self):
        # cleanup when the object is deleted
        self.cache.clear()
        del self.cache
