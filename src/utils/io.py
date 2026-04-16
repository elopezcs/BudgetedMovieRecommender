from __future__ import annotations

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_id(algo: str, seed: int, train_units: int) -> str:
    return f"{algo}_seed{seed}_steps{train_units}"


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def timestamp_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

