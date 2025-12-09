import concurrent.futures
from functools import partial
from pathlib import Path
from typing import List

from PIL import Image, ImageOps, ImageFile

# load corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_worker(file_path: Path, target_size=(768, 768), quality=85) -> Path:
    try:
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp']:
            return file_path

        with Image.open(file_path) as img:
            # Handle EXIF orientation (Images might not be in correct orientation)
            img = ImageOps.exif_transpose(img)

            if img.mode in ('RGBA', 'LA'):
                # handle transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background

            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Faster than .resize and handles aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            img.save(file_path, "JPEG", quality=quality, optimize=True)

        return file_path

    except Exception as e:
        return None


def resize_images_parallel(file_paths: List[Path], target_size=(768, 768), max_workers=None):
    # resize images using all available CPU cores
    worker_with_args = partial(resize_worker, target_size=target_size, quality=85)

    # uses all available CPU cores if max_workers is None
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker_with_args, file_paths))

    return [r for r in results if r is not None]
