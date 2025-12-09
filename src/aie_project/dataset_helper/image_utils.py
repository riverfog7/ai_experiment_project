import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Union, Tuple, List

from PIL import Image
from tqdm import tqdm

from .models import ImageDescription
from .utils import get_scale


class BatchCropper:
    def __init__(self,
                 output_dir: Union[str, Path],
                 num_workers: int = None,
                 size_threshold=(50,50),
                 pad: int = 5):
        # multithreaded image cropper
        # uses as many workers as CPU cores by default
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.size_threshold_x = size_threshold[0]
        self.size_threshold_y = size_threshold[1]
        self.pad = pad

    @staticmethod
    def _process_single_image(args: Tuple) -> List[Path | None]:
        # Image processing worker function (spawned in separate process)
        image_description, out_dir, thresh_x, thresh_y, pad = args
        if image_description.image_path is None:
            warnings.warn("Image path is None, skipping.")
            return []

        generated_files = []
        try:
            with Image.open(image_description.image_path) as img:
                # only get scale once
                scale = get_scale(image_description, img.size)
                for idx, obj in enumerate(image_description.annotation_info):
                    try:
                        # scale coordinates
                        x_min, y_min, x_max, y_max = obj.get_proper_bbox()
                        x_min *= scale
                        y_min *= scale
                        x_max *= scale
                        y_max *= scale

                        # apply padding
                        # padding is in percentage of the object size
                        box_width = x_max - x_min
                        box_height = y_max - y_min
                        x_min -= box_width * (pad / 100)
                        y_min -= box_height * (pad / 100)
                        x_max += box_width * (pad / 100)
                        y_max += box_height * (pad / 100)

                        x_min = int(max(0.0, x_min))
                        y_min = int(max(0.0, y_min))
                        x_max = int(min(img.width, x_max))
                        y_max = int(min(img.height, y_max))

                        # variable bounds checks
                        if x_max <= x_min or y_max <= y_min:
                            raise ValueError(f"Invalid bounding box after scaling: {(x_min, y_min, x_max, y_max)}")

                        if (x_max - x_min) < thresh_x and (y_max - y_min) < thresh_y:
                            raise ValueError(f"Cropped object size below threshold: {(x_max - x_min, y_max - y_min)}")

                        # crop and save
                        filename = f"{image_description.image_path.stem}_obj{idx}_{obj.class_name}.jpg"
                        save_path = out_dir / filename
                        cropped_img = img.crop((x_min, y_min, x_max, y_max))
                        cropped_img.save(save_path, format="JPEG", quality=90, optimize=True)
                        generated_files.append(save_path)

                    except Exception as e:
                        warnings.warn(f"Error processing object {idx} in {image_description.image_path}: {e}")
                        # append None (Maintains index alignment)
                        generated_files.append(None)

        except Exception as e:
            warnings.warn(f"Error processing {image_description.image_path}: {e}")
            return []

        return generated_files

    def process(self, descriptions: List['ImageDescription']) -> List[List[Path | None]]:
        # returns a list of lists of Paths (one list per image description)
        tasks = [(desc, self.output_dir, self.size_threshold_x, self.size_threshold_y, self.pad) for desc in descriptions]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # chunksize to reduce communication overhead
            chunksize = max(1, len(tasks) // (self.num_workers * 64))
            results = list(tqdm(
                executor.map(self._process_single_image, tasks, chunksize=chunksize),
                total=len(tasks),
                unit="img",
                desc="Cropping images"
            ))

        return results
