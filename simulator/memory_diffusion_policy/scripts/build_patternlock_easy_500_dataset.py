#!/usr/bin/env python3
"""Select, record, and finalize a leak-free 500-trajectory PatternLock Easy dataset.

The artifact keeps the 50 original Easy episodes at their benchmark episode IDs and adds
450 newly recorded episodes at IDs 100..549.  Candidate seeds are accepted only when their
complete ordered button pattern is absent from fresh_easy_100 and from every previously
accepted training configuration.

Run ``select`` and ``finalize`` with the DP environment.  Run ``record`` with the simulator
environment and the required Vulkan variables; it invokes the DP environment only for the
HDF5-to-Parquet conversion step.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = Path("/data/manu/robomme")
DEFAULT_OUT = DEFAULT_BASE / "patternlock_easy_500_v1"
TASK_TEXT = (
    "watch the video carefully, then use the stick attached to the robot to retrace the same pattern"
)


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as f:
        tmp = Path(f.name)
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def password_from_seed(seed: int) -> tuple[int, ...]:
    """Pure reimplementation of PatternLock Easy's deterministic seed-to-path draw."""
    import torch

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    adjacency = {}
    for row in range(3):
        for column in range(3):
            adjacency[row * 3 + column] = [
                (row + dr) * 3 + column + dc
                for dr, dc in directions
                if 0 <= row + dr < 3 and 0 <= column + dc < 3
            ]

    def randomized_dfs(start: int, target: int, generator: Any) -> list[int]:
        visited: set[int] = set()
        result: list[int] = []

        def visit(node: int, path: list[int]) -> bool:
            visited.add(node)
            path.append(node)
            if node == target:
                result.extend(path)
                return True
            neighbors = adjacency[node]
            order = torch.randperm(len(neighbors), generator=generator).tolist()
            for position in order:
                neighbor = neighbors[position]
                if neighbor not in visited and visit(neighbor, path):
                    return True
                if path and path[-1] != node:
                    path.pop()
            return False

        visit(start, [])
        return result

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    for _ in range(1000):
        start, end = torch.randperm(9, generator=generator)[:2].tolist()
        path = randomized_dfs(start, end, generator)
        if 2 <= len(path) <= 4:
            return tuple(map(int, path))
    raise RuntimeError(f"seed {seed}: Easy path draw did not terminate")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False) as f:
        tmp = Path(f.name)
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    tmp.replace(path)


def _source_path(dataset_root: Path, info: dict[str, Any], episode: int) -> Path:
    return dataset_root / info["data_path"].format(
        episode_chunk=int(episode) // int(info["chunks_size"]), episode_index=int(episode)
    )


def select(args: argparse.Namespace) -> int:
    out_root = args.out_root.resolve()
    selection_dir = out_root / "selection"
    dataset_root = out_root / "robomme_data_lerobot"
    selection_manifest = selection_dir / "manifest.json"
    custom_metadata = selection_dir / "record_dataset_PatternLock_metadata.json"
    if selection_manifest.exists() or custom_metadata.exists():
        raise FileExistsError(
            f"refusing to replace existing selection under {selection_dir}; use a new versioned root"
        )

    source_root = args.source_dataset_root.resolve()
    source_info = json.loads((source_root / "meta/info.json").read_text())
    memory_manifest = json.loads((args.source_memory_dir / "manifest.json").read_text())
    original_records: list[dict[str, Any]] = []
    used_patterns: set[tuple[int, ...]] = set()
    for record in sorted(memory_manifest["episodes"], key=lambda item: int(item["episode_index"])):
        episode = int(record["episode_index"])
        with np.load(args.source_memory_dir / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
            pattern = tuple(map(int, data["full_password_ids"]))
            seed = int(np.asarray(data["seed"]).item())
        original_records.append({
            "task": "PatternLock",
            "episode": episode,
            "seed": seed,
            "difficulty": "easy",
            "full_password": list(pattern),
            "source": "original_50",
        })
        used_patterns.add(pattern)
    if len(original_records) != 50:
        raise ValueError(f"expected 50 original Easy episodes, found {len(original_records)}")

    fresh_payload = json.loads(args.fresh_metadata.read_text())
    fresh_patterns = {tuple(map(int, record["full_password"])) for record in fresh_payload["records"]}
    if len(fresh_patterns) != 100:
        raise ValueError(f"fresh metadata must contain 100 distinct patterns, got {len(fresh_patterns)}")
    overlap = used_patterns.intersection(fresh_patterns)
    if overlap:
        raise ValueError(f"original training set overlaps fresh-100: {sorted(overlap)}")

    accepted: list[dict[str, Any]] = []
    rejection_counts = {"fresh100": 0, "training_duplicate": 0}
    candidate_index = 0
    while len(accepted) < args.num_new:
        seed = args.seed_start + args.seed_stride * candidate_index
        pattern = password_from_seed(seed)
        if pattern in fresh_patterns:
            rejection_counts["fresh100"] += 1
        elif pattern in used_patterns:
            rejection_counts["training_duplicate"] += 1
        else:
            episode = args.new_episode_start + len(accepted)
            accepted.append({
                "task": "PatternLock",
                "episode": episode,
                "seed": seed,
                "difficulty": "easy",
                "full_password": list(pattern),
                "candidate_index": candidate_index,
                "source": "generated_v1",
            })
            used_patterns.add(pattern)
        candidate_index += 1
        if candidate_index > args.max_candidates:
            raise RuntimeError(
                f"selected only {len(accepted)} additions from {candidate_index} candidates"
            )

    records = original_records + accepted
    if len(records) != 500 or len(accepted) != 450:
        raise AssertionError("v1 contract requires exactly 50 original + 450 new trajectories")
    final_overlap = {
        tuple(record["full_password"]) for record in records
    }.intersection(fresh_patterns)
    if final_overlap:
        raise AssertionError(f"selected training set overlaps fresh-100: {sorted(final_overlap)}")

    metadata_payload = {"env_id": "PatternLock", "record_count": len(records), "records": records}
    atomic_json(custom_metadata, metadata_payload)

    # Initialize the dedicated source tree with byte-identical copies of the original Easy files.
    (dataset_root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    for record in original_records:
        episode = int(record["episode"])
        src = _source_path(source_root, source_info, episode)
        dst = dataset_root / "data/chunk-000" / f"episode_{episode:06d}.parquet"
        if dst.exists():
            raise FileExistsError(f"refusing to replace {dst}")
        shutil.copy2(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise IOError(f"copy hash mismatch for episode {episode}")

    payload = {
        "artifact_id": "dataset:patternlock_easy_500_v1",
        "contract": {
            "total_trajectories": 500,
            "original_trajectories": 50,
            "new_trajectories": 450,
            "configuration_fingerprint": "complete ordered full_password including reset-owned first button",
            "fresh100_overlap_allowed": False,
            "new_training_duplicates_allowed": False,
        },
        "source_dataset_root": str(source_root),
        "source_memory_manifest_sha256": sha256_file(args.source_memory_dir / "manifest.json"),
        "fresh100_metadata": str(args.fresh_metadata.resolve()),
        "fresh100_metadata_sha256": sha256_file(args.fresh_metadata),
        "custom_metadata": str(custom_metadata),
        "custom_metadata_sha256": sha256_file(custom_metadata),
        "seed_search": {
            "start": args.seed_start,
            "stride": args.seed_stride,
            "candidates_examined": candidate_index,
            "rejection_counts": rejection_counts,
        },
        "counts": {
            "records": len(records),
            "distinct_full_passwords": len({tuple(r["full_password"]) for r in records}),
            "fresh100_overlap": 0,
            "new_records": len(accepted),
            "distinct_new_full_passwords": len({tuple(r["full_password"]) for r in accepted}),
        },
        "records": records,
    }
    atomic_json(selection_manifest, payload)
    print(json.dumps({key: payload[key] for key in ("artifact_id", "counts", "seed_search")}, indent=2))
    return 0


def _h5_string(value: Any) -> str:
    value = value.item() if isinstance(value, np.ndarray) and value.shape == () else value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def export_h5(h5_path: Path, episode: int, out_npz: Path) -> dict[str, Any]:
    import h5py

    def png_bytes(array: np.ndarray) -> bytes:
        output = BytesIO()
        Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(output, format="PNG")
        return output.getvalue()

    with h5py.File(h5_path, "r") as handle:
        group = handle[f"episode_{episode}"]
        keys = sorted(
            (key for key in group if key.startswith("timestep_")),
            key=lambda key: int(key.split("_")[1]),
        )
        if not keys:
            raise ValueError(f"{h5_path}: no recorded timesteps")
        front, wrist, states, actions, is_demo = [], [], [], [], []
        simple, grounded, simple_online, grounded_online = [], [], [], []
        # BPP_v2 row metadata (present only in bpp_v2 recordings; never policy conditioning)
        bpp_v2 = "bpp_v2_segment_kind" in group[keys[0]]["info"]
        v2_owner, v2_seg_id, v2_seg_kind, v2_accepted, v2_row = [], [], [], [], []
        for key in keys:
            row = group[key]
            if bpp_v2:
                v2_owner.append(_h5_string(row["info/bpp_v2_control_owner"][()]))
                v2_seg_id.append(int(row["info/bpp_v2_segment_id"][()]))
                v2_seg_kind.append(_h5_string(row["info/bpp_v2_segment_kind"][()]))
                v2_accepted.append(int(row["info/bpp_v2_accepted_event"][()]))
                v2_row.append(int(row["info/bpp_v2_row"][()]))
            front.append(png_bytes(row["obs/front_rgb"][...]))
            wrist.append(png_bytes(row["obs/wrist_rgb"][...]))
            state = np.asarray(row["obs/joint_state"][...], dtype=np.float32).reshape(-1)
            if state.shape != (7,):
                raise ValueError(f"{key}: expected seven stick joints, got {state.shape}")
            states.append(np.concatenate([state, np.zeros(1, dtype=np.float32)]))
            action = np.asarray(row["action/joint_action"][...], dtype=np.float32).reshape(-1)
            if action.shape != (8,):
                raise ValueError(f"{key}: expected eight action values, got {action.shape}")
            actions.append(action)
            is_demo.append(bool(row["info/is_video_demo"][()]))
            simple.append(_h5_string(row["info/simple_subgoal"][()]))
            grounded.append(_h5_string(row["info/grounded_subgoal"][()]))
            simple_online.append(_h5_string(row["info/simple_subgoal_online"][()]))
            grounded_online.append(_h5_string(row["info/grounded_subgoal_online"][()]))

    demo = np.asarray(is_demo, dtype=bool)
    non_demo = np.flatnonzero(~demo)
    if non_demo.size == 0:
        raise ValueError(f"episode {episode}: no execution rows")
    exec_start = int(non_demo[0])
    if not np.array_equal(demo, np.arange(len(demo)) < exec_start):
        raise ValueError(f"episode {episode}: demo rows are not one strict prefix")
    extra = {}
    if bpp_v2:
        if v2_row != list(range(len(keys))):
            raise ValueError(f"episode {episode}: bpp_v2 row index is not 0..T-1 (a row was dropped or duplicated)")
        extra = {
            "control_owner": np.asarray(v2_owner),
            "segment_id": np.asarray(v2_seg_id, dtype=np.int32),
            "segment_kind": np.asarray(v2_seg_kind),
            "accepted_event": np.asarray(v2_accepted, dtype=np.int32),
        }
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        front_png=np.asarray(front, dtype=object),
        wrist_png=np.asarray(wrist, dtype=object),
        state=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        is_demo=demo,
        simple_subgoal=np.asarray(simple),
        grounded_subgoal=np.asarray(grounded),
        simple_subgoal_online=np.asarray(simple_online),
        grounded_subgoal_online=np.asarray(grounded_online),
        **extra,
    )
    return {"num_frames": len(keys), "exec_start_idx": exec_start}


def record(args: argparse.Namespace) -> int:
    from tests._shared.dataset_generation import DatasetCase, _run_one_episode
    from robomme.env_record_wrapper import RobommeRecordWrapper

    # RecordWrapper currently places HDF5 buffering inside its video branch.  Keep that branch's
    # exact task filter active while dropping only the diagnostic video accumulation/encoding.
    # This leaves the raw HDF5 observations/actions unchanged and avoids a temporary MP4 per row.
    RobommeRecordWrapper._video_should_record = (
        lambda self, current_task_name: current_task_name != "NO RECORD"
    )
    RobommeRecordWrapper._video_append_step_frame = lambda self, *args, **kwargs: None
    RobommeRecordWrapper._video_flush_episode_files = lambda self, *args, **kwargs: None

    out_root = args.out_root.resolve()
    selection = json.loads((out_root / "selection/manifest.json").read_text())
    new_records = [record for record in selection["records"] if record["source"] == "generated_v1"]
    if args.episodes:
        wanted = set(map(int, args.episodes))
        new_records = [record for record in new_records if int(record["episode"]) in wanted]
        missing = sorted(wanted.difference(int(record["episode"]) for record in new_records))
        if missing:
            raise ValueError(f"requested episodes are not generated-v1 records: {missing}")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    new_records = [
        record for position, record in enumerate(new_records)
        if position % args.shard_count == args.shard_index
    ]

    template = args.template_parquet.resolve()
    converter = ROOT / "scripts/write_patternlock_episode_parquet.py"
    status_dir = out_root / "generation/status"
    scratch_root = out_root / "generation/scratch" / f"shard_{args.shard_index:02d}"
    parquet_dir = out_root / "robomme_data_lerobot/data/chunk-000"
    status_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for position, record_info in enumerate(new_records, start=1):
        episode = int(record_info["episode"])
        seed = int(record_info["seed"])
        expected_password = tuple(map(int, record_info["full_password"]))
        if password_from_seed(seed) != expected_password:
            raise ValueError(f"episode {episode}: seed/password selection drift")
        parquet = parquet_dir / f"episode_{episode:06d}.parquet"
        status_path = status_dir / f"episode_{episode:06d}.json"
        if parquet.exists() and status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("parquet_sha256") == sha256_file(parquet):
                print(f"[record-500] episode {episode}: reused ({position}/{len(new_records)})", flush=True)
                continue

        work = scratch_root / f"episode_{episode:06d}"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        case = DatasetCase(
            env_id="PatternLock",
            episode=episode,
            base_seed=seed,
            difficulty="easy",
            save_video=False,
            mode_tag="password_counter_500_v1",
        )
        try:
            if not _run_one_episode(case, seed, work):
                raise RuntimeError("scripted recorder did not terminate successfully")
            h5_path = work / "hdf5_files" / f"PatternLock_ep{episode}_seed{seed}.h5"
            intermediate = work / f"episode_{episode:06d}.npz"
            exported = export_h5(h5_path, episode, intermediate)
            converter_status = work / "converter_status.json"
            subprocess.run([
                str(args.converter_python), str(converter),
                "--input", str(intermediate),
                "--template", str(template),
                "--output", str(parquet),
                "--episode", str(episode),
                "--status-out", str(converter_status),
            ], check=True)
            converted = json.loads(converter_status.read_text())
            if converted["num_frames"] != exported["num_frames"]:
                raise ValueError("converter/export frame counts differ")
            status = {
                "status": "PASS",
                "episode": episode,
                "seed": seed,
                "full_password": list(expected_password),
                "num_frames": exported["num_frames"],
                "exec_start_idx": exported["exec_start_idx"],
                "parquet": str(parquet),
                "parquet_sha256": sha256_file(parquet),
            }
            atomic_json(status_path, status)
            shutil.rmtree(work)
            print(
                f"[record-500] episode {episode}: PASS T={status['num_frames']} "
                f"({position}/{len(new_records)})", flush=True,
            )
        except BaseException as exc:
            failure = {"episode": episode, "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            atomic_json(status_dir / f"episode_{episode:06d}.failed.json", failure)
            print(f"[record-500] episode {episode}: ERROR {failure['error']}", file=sys.stderr, flush=True)
            break
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2), file=sys.stderr)
        return 1
    return 0


def finalize(args: argparse.Namespace) -> int:
    out_root = args.out_root.resolve()
    dataset_root = out_root / "robomme_data_lerobot"
    selection_path = out_root / "selection/manifest.json"
    selection = json.loads(selection_path.read_text())
    source_root = Path(selection["source_dataset_root"])
    source_episodes = {
        int(record["episode_index"]): record
        for record in read_jsonl(source_root / "meta/episodes.jsonl")
    }
    episodes = []
    artifact_records = []
    total_frames = 0
    for record in sorted(selection["records"], key=lambda item: int(item["episode"])):
        episode = int(record["episode"])
        parquet = dataset_root / "data/chunk-000" / f"episode_{episode:06d}.parquet"
        if not parquet.exists():
            raise FileNotFoundError(f"episode {episode}: missing {parquet}")
        if record["source"] == "original_50":
            length = int(source_episodes[episode]["length"])
        else:
            status_path = out_root / "generation/status" / f"episode_{episode:06d}.json"
            status = json.loads(status_path.read_text())
            if status.get("status") != "PASS" or status.get("parquet_sha256") != sha256_file(parquet):
                raise ValueError(f"episode {episode}: generation status/hash gate failed")
            length = int(status["num_frames"])
        episodes.append({"episode_index": episode, "tasks": [TASK_TEXT], "length": length})
        artifact_records.append({
            "episode_index": episode,
            "seed": int(record["seed"]),
            "full_password": list(map(int, record["full_password"])),
            "source": record["source"],
            "num_frames": length,
            "parquet_sha256": sha256_file(parquet),
        })
        total_frames += length
    if len(episodes) != 500:
        raise ValueError(f"final dataset must contain 500 episodes, got {len(episodes)}")

    meta = dataset_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    source_info = json.loads((source_root / "meta/info.json").read_text())
    source_info.update({
        "total_episodes": 500,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": "0:500"},
    })
    atomic_json(meta / "info.json", source_info)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": TASK_TEXT}])
    write_jsonl(meta / "episodes.jsonl", episodes)

    fresh_patterns = {
        tuple(record["full_password"])
        for record in json.loads(Path(selection["fresh100_metadata"]).read_text())["records"]
    }
    training_patterns = {tuple(record["full_password"]) for record in artifact_records}
    overlap = sorted(training_patterns.intersection(fresh_patterns))
    if overlap:
        raise AssertionError(f"finalized dataset overlaps fresh-100: {overlap}")
    manifest = {
        "artifact_id": "dataset:patternlock_easy_500_v1",
        "status": "PASS",
        "dataset_root": str(dataset_root),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "fresh100_metadata": selection["fresh100_metadata"],
        "fresh100_metadata_sha256": selection["fresh100_metadata_sha256"],
        "num_episodes": len(artifact_records),
        "num_frames": total_frames,
        "num_distinct_full_passwords": len(training_patterns),
        "fresh100_full_password_overlap": 0,
        "episodes": artifact_records,
    }
    atomic_json(out_root / "dataset_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in (
        "status", "num_episodes", "num_frames", "num_distinct_full_passwords",
        "fresh100_full_password_overlap")}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-root", type=Path, default=DEFAULT_OUT)

    p = sub.add_parser("select", parents=[common])
    p.add_argument("--source-dataset-root", type=Path, default=DEFAULT_BASE / "robomme_data_lerobot")
    p.add_argument("--source-memory-dir", type=Path, default=DEFAULT_BASE / "cache/patternlock_easy_memory_v1")
    p.add_argument(
        "--fresh-metadata", type=Path,
        default=ROOT / "eval_envs/env_lists/fresh_easy_100/record_dataset_PatternLock_metadata.json",
    )
    p.add_argument("--num-new", type=int, default=450)
    p.add_argument("--new-episode-start", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=3_000_000)
    p.add_argument("--seed-stride", type=int, default=100)
    p.add_argument("--max-candidates", type=int, default=100_000)
    p.set_defaults(func=select)

    p = sub.add_parser("record", parents=[common])
    p.add_argument("--episodes", type=int, nargs="*")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--converter-python", type=Path, default=Path("/data/manu/envs/robomme_dp/bin/python"))
    p.add_argument(
        "--template-parquet", type=Path,
        default=DEFAULT_BASE / "robomme_data_lerobot/data/chunk-000/episode_000000.parquet",
    )
    p.set_defaults(func=record)

    p = sub.add_parser("finalize", parents=[common])
    p.set_defaults(func=finalize)
    return ap


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
