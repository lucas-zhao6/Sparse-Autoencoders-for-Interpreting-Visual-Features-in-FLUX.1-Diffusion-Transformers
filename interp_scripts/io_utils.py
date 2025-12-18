# io_utils.py
import os
import json
from typing import Any, Dict, List, Tuple


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def read_prompts(path: str) -> List[str]:
    """Read prompts from text file, one per line."""
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines()]
    prompts = [p for p in prompts if p]  # Remove empty lines
    return prompts


def write_json(path: str, obj: Any) -> None:
    """Write object to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str) -> Any:
    """Read JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    """Append a row to JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all rows from JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest(manifest_path: str, out_dir: str) -> Tuple[List[Dict], Dict[int, Dict]]:
    """
    Load manifest JSONL and return sorted records + lookup dict.
    
    Returns:
        records: List of dicts sorted by image_id
        id_to_record: Dict mapping image_id to record
    """
    records = read_jsonl(manifest_path)
    
    for r in records:
        r["_image_abs_path"] = os.path.join(out_dir, r["image_path"])
    
    records.sort(key=lambda r: int(r["image_id"]))
    id_to_record = {int(r["image_id"]): r for r in records}
    
    return records, id_to_record


def count_manifest_rows(manifest_path: str) -> int:
    """Count number of rows in manifest file."""
    n = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
