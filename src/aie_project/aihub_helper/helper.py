import os
import concurrent.futures
from pathlib import Path
from typing import Union, Optional, List, Callable

import requests
from pydantic import SecretStr
from tqdm import tqdm

from .models import AIHubDataset
from .utils import parse_aihub_tree, extract_and_merge, unzip_file


class AIHubHelper:
    def __init__(
            self,
            api_key: Optional[str] = None,
            user_agent: str = "curl/7.68.0",
            max_background_workers: int = 4
    ):
        if api_key is None:
            api_key = os.getenv("AIHUB_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key must be provided either as an argument or through the AIHUB_API_KEY environment variable.")
        self.api_key = SecretStr(api_key)
        self.user_agent = user_agent

        # We do not use processpool here because CPU heavy operations are internally done
        # With a process pool. See image_utils.py for more details.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_background_workers)

    def get_api_key(self) -> str:
        return self.api_key.get_secret_value()

    def get_auth_header(self) -> dict:
        return {"apikey": self.api_key.get_secret_value()}

    def list_dataset(self, dataset_key: Union[str, int], is_package: bool = False) -> AIHubDataset:
        # No credentials needed for this endpoint
        url = f"https://api.aihub.or.kr/info/pckage/{dataset_key}.do" if is_package else f"https://api.aihub.or.kr/info/{dataset_key}.do"
        response = requests.get(url, headers={"User-Agent": self.user_agent})

        if response.status_code != 200:
            # This is correct. For some reason, the API returns 502 with a correct response.
            if response.status_code == 502 and "The contents are encoded" in response.text:
                pass
            else:
                raise ValueError(f"Wrong status code received from AIHub API: {response.status_code}")

        return parse_aihub_tree(response.text)

    def download_file(
            self,
            dataset_key: Union[str, int],
            file_sn: Union[str, int, List[Union[str, int]]] = "all",
            output_file: Path = Path("./download.tar"),
            is_package: bool = False,
    ) -> Path:
        base_url = "https://api.aihub.or.kr/down/pckage/0.6" if is_package else "https://api.aihub.or.kr/down/0.6"
        url = f"{base_url}/{dataset_key}.do"
        if isinstance(file_sn, list):
            file_sn = ",".join(map(str, file_sn))

        headers = self.get_auth_header()
        params = {"fileSn": str(file_sn)}

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_file.resolve()

        tqdm.write(f"Downloading file_sn={file_sn}...")

        with requests.get(url, headers=headers, params=params, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(output_file.as_posix(), 'wb') as f:
                # Tqdm progress bar with total size. Can be nested in another tqdm so set leave=False.
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading", leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        return output_file

    def _post_process_pipeline(
            self,
            tar_path: Path,
            output_dir: Path,
            unzip: bool,
            create_zipfile_directory: bool,
            transform: Optional[Callable[[List[Path]], None]]
    ):
        try:
            # This function merges extracts the tarball and merges .part* files.
            merged = extract_and_merge(tar_path=tar_path, dest_dir=output_dir)
            if tar_path.exists():
                tar_path.unlink()

            extracted_files = []
            if unzip:
                for file_path in merged:
                    # Unzip all files that are zip files. Delete them after extraction to save space.
                    # Returns zip file contents.
                    if file_path.suffix.lower() == '.zip':
                        unzipped = unzip_file(
                            file_path,
                            delete_zip=True,
                            create_directory=create_zipfile_directory
                        )
                        extracted_files.extend(unzipped)
                    else:
                        extracted_files.append(file_path)
            else:
                extracted_files = merged

            if transform and extracted_files:
                # Execute user-defined transform logic with file list as arguments.
                transform(extracted_files)

            tqdm.write(f"Background processing complete for {tar_path.name}")

        except Exception as e:
            tqdm.write(f"Error during background processing of {tar_path.name}: {e}")

    def download_and_extract_file(
            self,
            dataset_key: Union[str, int],
            file_sn: Union[str, int, List[Union[str, int]]] = "all",
            tmp_dir: Path = Path("./.temp"),
            output_dir: Path = Path("./extracted"),
            is_package: bool = False,
            unzip: bool = True,
            create_zipfile_directory: bool = True,
            transform: Optional[Callable[[List[Path]], None]] = None,
            background_processing: bool = False
    ) -> Path:
        # A convenience function that downloads, extracts, unzips and transforms files.
        tmp_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        sn_str = str(file_sn) if not isinstance(file_sn, list) else "multi_" + "_".join(map(str, file_sn))
        download_path = tmp_dir / f"dataset_{dataset_key}_{sn_str}.tar"

        self.download_file(
            dataset_key=dataset_key,
            file_sn=file_sn,
            output_file=download_path,
            is_package=is_package,
        )

        if background_processing:
            # Run post processing in the background if requested.
            self._executor.submit(
                self._post_process_pipeline,
                tar_path=download_path,
                output_dir=output_dir,
                unzip=unzip,
                create_zipfile_directory=create_zipfile_directory,
                transform=transform
            )
            tqdm.write(f"Sent {sn_str} to background processing...")
        else:
            self._post_process_pipeline(
                tar_path=download_path,
                output_dir=output_dir,
                unzip=unzip,
                create_zipfile_directory=create_zipfile_directory,
                transform=transform
            )

        return output_dir

    def shutdown(self):
        self._executor.shutdown(wait=True)
