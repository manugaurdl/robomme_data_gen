#!/usr/bin/env python3
"""Convert one recorder export into a schema-identical RoboMME LeRobot parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TASK_INDEX = 0

# BPP_v2 row metadata columns, appended after the v1 template columns when the export carries them (never conditioning).
BPP_V2_COLUMNS = (
    ("control_owner", pa.string(), "string"),
    ("segment_id", pa.int32(), "int32"),
    ("segment_kind", pa.string(), "string"),
    ("accepted_event", pa.int32(), "int32"),
)


def _bpp_v2_schema(schema: pa.Schema) -> pa.Schema:
    fields = list(schema) + [pa.field(name, typ) for name, typ, _ in BPP_V2_COLUMNS]
    metadata = dict(schema.metadata or {})
    if b"huggingface" in metadata:
        hf = json.loads(metadata[b"huggingface"])
        features = hf.get("info", {}).get("features") if isinstance(hf.get("info"), dict) else hf.get("features")
        if isinstance(features, dict):
            for name, _, dtype in BPP_V2_COLUMNS:
                features[name] = {"dtype": dtype, "_type": "Value"}
        metadata[b"huggingface"] = json.dumps(hf).encode()
    return pa.schema(fields, metadata=metadata)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--status-out", type=Path, required=True)
    args = ap.parse_args()

    with np.load(args.input, allow_pickle=True) as data:
        values = {key: data[key] for key in data.files}
    state = np.asarray(values["state"], dtype=np.float32)
    actions = np.asarray(values["actions"], dtype=np.float32)
    demo = np.asarray(values["is_demo"], dtype=bool)
    num_frames = len(demo)
    if state.shape != (num_frames, 8) or actions.shape != (num_frames, 8):
        raise ValueError(f"numeric shape mismatch state={state.shape} action={actions.shape} T={num_frames}")
    non_demo = np.flatnonzero(~demo)
    if non_demo.size == 0:
        raise ValueError("episode contains no execution rows")
    exec_start = int(non_demo[0])
    if not np.array_equal(demo, np.arange(num_frames) < exec_start):
        raise ValueError("is_demo is not one strict prefix")

    schema = pq.read_schema(args.template)
    bpp_v2 = "segment_kind" in values
    if bpp_v2:
        schema = _bpp_v2_schema(schema)
    frame = np.arange(num_frames, dtype=np.int64)
    scalar_columns = {
        "exec_start_idx": np.full(num_frames, exec_start, dtype=np.int32),
        "is_demo": demo,
        "step_idx": frame.astype(np.int32),
        "epis_idx": np.full(num_frames, args.episode, dtype=np.int32),
        "timestamp": (frame / 10.0).astype(np.float32),
        "frame_index": frame,
        "episode_index": np.full(num_frames, args.episode, dtype=np.int64),
        "index": np.asarray(args.episode * 1_000_000 + frame, dtype=np.int64),
        "task_index": np.full(num_frames, TASK_INDEX, dtype=np.int64),
    }
    source = {
        "image": [{"bytes": bytes(value), "path": None} for value in values["front_png"]],
        "wrist_image": [{"bytes": bytes(value), "path": None} for value in values["wrist_png"]],
        "state": state.tolist(),
        "actions": actions.tolist(),
        "simple_subgoal": values["simple_subgoal"].tolist(),
        "grounded_subgoal": values["grounded_subgoal"].tolist(),
        "simple_subgoal_online": values["simple_subgoal_online"].tolist(),
        "grounded_subgoal_online": values["grounded_subgoal_online"].tolist(),
        **scalar_columns,
    }
    if bpp_v2:
        source.update({
            "control_owner": [str(v) for v in values["control_owner"].tolist()],
            "segment_id": np.asarray(values["segment_id"], dtype=np.int32),
            "segment_kind": [str(v) for v in values["segment_kind"].tolist()],
            "accepted_event": np.asarray(values["accepted_event"], dtype=np.int32),
        })
        for key in ("control_owner", "segment_id", "segment_kind", "accepted_event"):
            if len(source[key]) != num_frames:
                raise ValueError(f"bpp_v2 column {key} has {len(source[key])} rows, expected {num_frames}")
    arrays = []
    for field in schema:
        if field.name not in source:
            raise ValueError(f"template column {field.name!r} has no converter source")
        arrays.append(pa.array(source[field.name], type=field.type))
    table = pa.Table.from_arrays(arrays, schema=schema)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=args.output.name + ".", delete=False) as f:
        tmp = Path(f.name)
    try:
        pq.write_table(table, tmp, compression="snappy")
        check = pq.read_table(tmp)
        if check.schema != schema or check.num_rows != num_frames:
            raise ValueError("written parquet failed schema/row-count round trip")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(args.output)
    finally:
        if tmp.exists():
            tmp.unlink()

    status = {
        "status": "PASS",
        "episode": args.episode,
        "num_frames": num_frames,
        "exec_start_idx": exec_start,
        "parquet_sha256": sha256_file(args.output),
        "bpp_v2_columns": bool(bpp_v2),
    }
    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    args.status_out.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
