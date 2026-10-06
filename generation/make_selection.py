"""Select fresh disjoint PatternLock Easy seeds for small or medium data runs."""

from __future__ import annotations

import argparse
from functools import cache
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "simulator/memory_diffusion_policy/scripts/build_patternlock_easy_500_dataset.py"


@cache
def _password_draw():
    spec = importlib.util.spec_from_file_location("patternlock_seed_draw", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.password_from_seed


def password_from_seed(seed: int) -> tuple[int, ...]:
    return _password_draw()(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--dev48", type=int, required=True)
    parser.add_argument("--fresh100", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=15001)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=100000)
    args = parser.parse_args()
    counts = {"train": args.train, "dev48": args.dev48, "fresh100": args.fresh100}
    if any(n < 0 for n in counts.values()) or args.seed_stride <= 0 or args.max_candidates <= 0:
        parser.error("counts must be nonnegative; stride and max-candidates must be positive")
    paths = {split: args.out_root / split / "selection" for split in counts}
    for split, path in paths.items():
        if counts[split] and (path / "manifest.json").exists():
            raise FileExistsError(path / "manifest.json")

    used: set[tuple[int, ...]] = set()
    selected = {split: [] for split in counts}
    candidate = 0
    for split, count in counts.items():
        while len(selected[split]) < count:
            if candidate >= args.max_candidates:
                raise RuntimeError("not enough distinct Easy passwords; lower counts or design explicit route variants")
            seed = args.seed_start + args.seed_stride * candidate
            candidate += 1
            password = password_from_seed(seed)
            if password in used:
                continue
            used.add(password)
            selected[split].append({"task": "PatternLock", "episode": len(selected[split]),
                                    "seed": seed, "difficulty": "easy", "full_password": list(password),
                                    "source": "new_seed_v1"})

    for split, records in selected.items():
        if not records:
            continue
        path = paths[split]
        path.mkdir(parents=True, exist_ok=True)
        payload = {"env_id": "PatternLock", "record_count": len(records), "records": records}
        for name in ("manifest.json", "record_dataset_PatternLock_metadata.json"):
            (path / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": {key: len(value) for key, value in selected.items()},
                      "candidates_checked": candidate, "out_root": str(args.out_root)}))


if __name__ == "__main__":
    main()
