import os
from pathlib import Path
from typing import Union, Optional, List

import requests
from pydantic import SecretStr
from tqdm import tqdm

from .models import AIHubDataset
from .utils import parse_aihub_tree, extract_and_merge, unzip_file


class AIHubHelper:
    def __init__(self, api_key: Optional[str] = None, user_agent: str = "curl/7.68.0"):
        if api_key is None:
            api_key = os.getenv("AIHUB_API_KEY")
            if not api_key:
                raise ValueError("API key must be provided either as an argument or through the AIHUB_API_KEY environment variable.")
        self.api_key = SecretStr(api_key)
        self.user_agent = user_agent

    def get_api_key(self) -> str:
        return self.api_key.get_secret_value()

    def get_auth_header(self) -> dict:
        return {"apikey": self.api_key.get_secret_value()}

    def list_dataset(self, dataset_key: Union[str, int], is_package: bool = False) -> AIHubDataset:
        url = f"https://api.aihub.or.kr/info/pckage/{dataset_key}.do" if is_package else f"https://api.aihub.or.kr/info/{dataset_key}.do"
        response = requests.get(url, headers={"User-Agent": self.user_agent})
        if not response.status_code == 502:
            # this is correct. The API somehow returns 502 for valid requests
            raise ValueError(f"Wrong status code received from AIHub API: {response.status_code}")

        return parse_aihub_tree(response.text)

    def download_file(
            self,
            dataset_key: Union[str, int],
            file_sn: Union[str, int, List[Union[str, int]]] = "all",
            output_file: Path = Path("./download.tar"),
            is_package: bool = False,
    ) -> Path:
        # Determine the correct base URL based on whether it's a package or a standard dataset
        base_url = "https://api.aihub.or.kr/down/pckage/0.6" if is_package else "https://api.aihub.or.kr/down/0.6"
        url = f"{base_url}/{dataset_key}.do"
        if isinstance(file_sn, list):
            file_sn = ",".join(map(str, file_sn))

        headers = self.get_auth_header()
        params = {"fileSn": str(file_sn)}

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_file.resolve()
        tqdm.write(f"Downloading file_sn={file_sn} from dataset={dataset_key}...")

        with requests.get(url, headers=headers, params=params, stream=True) as response:
            response.raise_for_status()
            # Get total size from headers for the progress bar
            total_size = int(response.headers.get('content-length', 0))

            with open(output_file.as_posix(), 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading", leave=False) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

        return output_file

    def download_and_extract_file(
            self,
            dataset_key: Union[str, int],
            file_sn: Union[str, int, List[Union[str, int]]] = "all",
            tmp_dir: Path = Path("./.temp"),
            output_dir: Path = Path("./extracted"),
            is_package: bool = False,
            unzip: bool = True,
            create_zipfile_directory: bool = True,
            transform: Optional[callable] = None,
    ) -> Path:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        download_path = tmp_dir / f"dataset_{dataset_key}_files.tar"
        self.download_file(
            dataset_key=dataset_key,
            file_sn=file_sn,
            output_file=download_path,
            is_package=is_package,
        )
        merged = extract_and_merge(tar_path=download_path, dest_dir=output_dir)

        extracted_files = []
        if unzip:
            for file_path in merged:
                if file_path.suffix.lower() == '.zip':
                    unzipped = unzip_file(file_path, delete_zip=True, create_directory=create_zipfile_directory)
                    extracted_files.extend(unzipped)

        if transform and extracted_files:
            transform(extracted_files)

        return output_dir
