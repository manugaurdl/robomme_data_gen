from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start_file: str | Path) -> Path:
    path = Path(start_file).resolve()
    cur = path if path.is_dir() else path.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root from {path}")


def ensure_src_on_path(start_file: str | Path) -> Path:
    repo_root = find_repo_root(start_file)
    src_path = repo_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return repo_root

