from __future__ import annotations

from pathlib import Path


def latest_run_dir(base: str | Path) -> Path:
    root = Path(base)
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found under {root}")
    return sorted(runs)[-1]

