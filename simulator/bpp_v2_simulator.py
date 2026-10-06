"""BPP_v2 simulator lock (replaces roboMME/data_gen/simulator.py's 050f770-only lock for the v2 tree).

The repo lock (`roboMME/data_gen/simulator.py::validate_simulator`) refuses any checkout that is not a clean git checkout
of 050f770 -- by design it rejects the BPP_v2 tree. This module is the explicit v2 replacement: it pins the v2 source
tree by content hash instead of by git revision.

  python bpp_v2_simulator.py write  [ROOT]   # (re)write ROOT/bpp_v2_simulator.lock.json from the current tree
  python bpp_v2_simulator.py check  [ROOT]   # verify the tree against the lock (exit 1 on any mismatch)

Library: `validate_simulator_v2(root)` -> dict (raises ValueError on mismatch); `load_recording_api_v2(root)` imports the
scripted collector (`tests._shared.dataset_generation`) from the validated tree only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

LOCK_NAME = "bpp_v2_simulator.lock.json"
BASE_REVISION = "050f770ec40a37dd39e5ee0e5229ecab98dc2e27"
BASE_MANIFEST_SHA256_NOTE = ".bpp_v1/setup/src050_manifest.sha256 (382 files) = the pristine base of this tree"
MANISKILL_FORK = "YinpeiDai/ManiSkill@07be6fbc66350ddca200abfb0a11b692f078f7fd (unmodified; from the BPP_v1 venv)"
RUNTIME_FILES = (
    "robomme_benchmark/src/robomme/robomme_env/PatternLock.py",
    "robomme_benchmark/src/robomme/robomme_env/utils/bpp_v2.py",
    "robomme_benchmark/src/robomme/robomme_env/utils/subgoal_planner_func.py",
    "robomme_benchmark/src/robomme/robomme_env/utils/subgoal_evaluate_func.py",
    "robomme_benchmark/src/robomme/robomme_env/utils/planner_fail_safe.py",
    "robomme_benchmark/src/robomme/robomme_env/utils/planner_denseStep.py",
    "robomme_benchmark/src/robomme/robomme_env/utils/statechange.py",
    "robomme_benchmark/src/robomme/env_record_wrapper/RecordWrapper.py",
    "robomme_benchmark/src/robomme/env_record_wrapper/DemonstrationWrapper.py",
    "robomme_benchmark/src/robomme/env_record_wrapper/FailAwareWrapper.py",
    "robomme_benchmark/src/robomme/env_record_wrapper/episode_config_resolver.py",
    "robomme_benchmark/tests/_shared/dataset_generation.py",
    "memory_diffusion_policy/scripts/build_patternlock_easy_500_dataset.py",
    "memory_diffusion_policy/scripts/write_patternlock_episode_parquet.py",
)
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(root: Path) -> dict:
    root = Path(root).resolve()
    files = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or EXCLUDE_PARTS.intersection(p.relative_to(root).parts) or p.name == LOCK_NAME:
            continue
        if p.suffix in (".pyc", ".pyo"):
            continue
        files[p.relative_to(root).as_posix()] = _sha(p)
    return files


def _manifest_digest(files: dict) -> str:
    return hashlib.sha256("".join(f"{v}  ./{k}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def write_lock(root: Path) -> dict:
    root = Path(root).resolve()
    files = tree_manifest(root)
    lock = {
        "schema": "bpp_v2_simulator_lock_v1",
        "base_revision": BASE_REVISION,
        "base_note": BASE_MANIFEST_SHA256_NOTE,
        "maniskill": MANISKILL_FORK,
        "env_flag": {"gym.make kwarg": "bpp_v2", "value": True},
        "runtime_files_sha256": {name: files[name] for name in RUNTIME_FILES},
        "tree_files": len(files),
        "tree_manifest_sha256": _manifest_digest(files),
        "tree_files_sha256": files,
    }
    (root / LOCK_NAME).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    return lock


def validate_simulator_v2(root: Path) -> dict:
    root = Path(root).resolve()
    lock_path = root / LOCK_NAME
    lock = json.loads(lock_path.read_text())
    files = tree_manifest(root)
    for name, expected in lock["runtime_files_sha256"].items():
        if files.get(name) != expected:
            raise ValueError(f"bpp_v2 simulator runtime file differs from lock: {name}")
    digest = _manifest_digest(files)
    if digest != lock["tree_manifest_sha256"]:
        changed = sorted(set(files.items()) ^ set(lock["tree_files_sha256"].items()))
        raise ValueError(f"bpp_v2 simulator tree differs from lock ({len(changed)} entries), e.g. {changed[:4]}")
    return {"root": str(root), "lock_sha256": _sha(lock_path), "tree_manifest_sha256": digest,
            "runtime_files_sha256": dict(lock["runtime_files_sha256"])}


def load_recording_api_v2(root: Path):
    """Validate the tree, then import the scripted collector from it (never from anywhere else)."""
    info = validate_simulator_v2(root)
    root = Path(root).resolve()
    benchmark = root / "robomme_benchmark"
    source_root = (benchmark / "src/robomme").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "robomme" or name.startswith("robomme."):
            path = getattr(module, "__file__", None)
            if path is not None and source_root not in Path(path).resolve().parents:
                raise ImportError(f"{name} was loaded outside the bpp_v2 simulator tree: {path}")
    for path in (benchmark, benchmark / "src"):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    for name in tuple(sys.modules):
        if name == "tests" or name.startswith("tests."):
            del sys.modules[name]
    module = importlib.import_module("tests._shared.dataset_generation")
    if Path(module.__file__).resolve() != (benchmark / "tests/_shared/dataset_generation.py").resolve():
        raise ImportError(f"collector resolved outside the bpp_v2 tree: {module.__file__}")
    return module, info


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    root = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent)
    if cmd == "write":
        lock = write_lock(root)
        print(json.dumps({k: lock[k] for k in ("tree_files", "tree_manifest_sha256")}))
    else:
        try:
            print(json.dumps(validate_simulator_v2(root), indent=1))
        except ValueError as exc:
            print(f"LOCK MISMATCH: {exc}")
            raise SystemExit(1)
