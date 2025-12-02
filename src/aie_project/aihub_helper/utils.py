import re
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Union, List

from .models import AIHubDataset, AIHubFile, AIHubFolder


def parse_aihub_tree(raw_text: str) -> AIHubDataset:
    # Parse API response (A file tree)
    tree_char_re = re.compile(r'[└├]')
    file_info_re = re.compile(r'\s*\|\s*')

    root_list: List[Union[AIHubFile, AIHubFolder]] = []
    stack = [{'indent': -1, 'list': root_list}]

    for line in raw_text.split('\n'):
        if not line.strip() or "====" in line or "encoded" in line:
            # skip unnecessary segments
            continue

        match = tree_char_re.search(line)
        if not match:
            continue

        current_indent = match.start()
        content_raw = line[current_indent:].lstrip('└├│─ ')

        node: Union[AIHubFile, AIHubFolder]

        if '|' in content_raw:
            parts = file_info_re.split(content_raw)
            if len(parts) < 3:
                continue

            node = AIHubFile(
                name=parts[0].strip(),
                size=parts[1].strip(),
                file_sn=parts[2].strip()
            )
        else:
            node = AIHubFolder(
                name=content_raw.strip(),
                children=[]
            )

        while len(stack) > 1 and current_indent <= stack[-1]['indent']:
            stack.pop()

        stack[-1]['list'].append(node)

        if isinstance(node, AIHubFolder):
            stack.append({'indent': current_indent, 'list': node.children})

    if not root_list:
        raise ValueError("Parsed result is empty.")

    if len(root_list) > 1:
        raise ValueError(f"Expected exactly 1 root folder, found {len(root_list)}")

    root_node = root_list[0]

    if not isinstance(root_node, AIHubFolder):
        raise ValueError("Root element must be a folder.")

    return AIHubDataset(
        dataset_name=root_node.name,
        root_folder=root_node
    )


def extract_and_merge(tar_path: Union[str, Path], dest_dir: Union[str, Path]) -> List[Path]:
    tar_path = Path(tar_path)
    dest_dir = Path(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(dest_dir)

    merge_tasks = defaultdict(list)
    part_pattern = re.compile(r'(.+)\.part(\d+)$')

    for file_path in dest_dir.rglob("*"):
        if file_path.is_file():
            match = part_pattern.match(file_path.name)
            if match:
                base_name = match.group(1)
                offset = int(match.group(2))
                target_file = file_path.parent / base_name
                merge_tasks[target_file].append((offset, file_path))

    merged_files = []

    for target_file, parts in merge_tasks.items():
        parts.sort(key=lambda x: x[0])

        with open(target_file, 'wb') as outfile:
            for _, part_path in parts:
                with open(part_path, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
                part_path.unlink()

        merged_files.append(target_file)

    return merged_files


def unzip_file(zip_path: Union[str, Path], dest_dir: Union[str, Path] = None, delete_zip: bool = True, create_directory: bool = True) -> List[Path]:
    zip_path = Path(zip_path)
    if dest_dir is None:
        dest_dir = zip_path.parent
        if create_directory:
            dest_dir = dest_dir / zip_path.stem
    else:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

    extracted_paths = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        for name in zip_ref.namelist():
            clean_name = name.lstrip("/\\")
            full_path = dest_dir / clean_name
            extracted_paths.append(full_path)

    if delete_zip:
        zip_path.unlink()

    return extracted_paths
