from pathlib import Path
from typing import List, Dict
import json


def get_files_to_handle(
    path_a: Path, path_b: Path, extension_a: str, extension_b: str
) -> List[str]:
    """
    Find files in path_a that are not in path_b
    """
    files_a = path_a.glob(f"*.{extension_a}")
    files_b = path_b.glob(f"*.{extension_b}")
    files_a_stem = [file.stem for file in files_a]
    files_b_stem = [file.stem for file in files_b]
    files_to_handle = list(set(files_a_stem) - set(files_b_stem))
    return [f"{file}.{extension_a}" for file in files_to_handle]


def save_json_to_path(data: Dict, path: Path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json_from_path(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)
