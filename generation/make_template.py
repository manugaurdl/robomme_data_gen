"""Create an empty PatternLock Parquet schema template for the episode writer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    image = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    vector = pa.list_(pa.float32(), 8)
    types = [
        ("image", image), ("wrist_image", image), ("state", vector), ("actions", vector),
        ("exec_start_idx", pa.int32()), ("is_demo", pa.bool_()), ("step_idx", pa.int32()),
        ("epis_idx", pa.int32()), ("simple_subgoal", pa.string()),
        ("grounded_subgoal", pa.string()), ("simple_subgoal_online", pa.string()),
        ("grounded_subgoal_online", pa.string()), ("timestamp", pa.float32()),
        ("frame_index", pa.int64()), ("episode_index", pa.int64()),
        ("index", pa.int64()), ("task_index", pa.int64()),
    ]
    features = {}
    for name, dtype in types:
        if name in ("image", "wrist_image"):
            features[name] = {"_type": "Image"}
        elif name in ("state", "actions"):
            features[name] = {"feature": {"dtype": "float32", "_type": "Value"},
                              "length": 8, "_type": "Sequence"}
        else:
            features[name] = {"dtype": str(dtype), "_type": "Value"}
    schema = pa.schema([pa.field(name, dtype) for name, dtype in types],
                       metadata={b"huggingface": json.dumps({"info": {"features": features}}).encode()})
    table = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out, compression="snappy")
    assert pq.read_schema(args.out) == schema
    print(json.dumps({"path": str(args.out), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "fields": len(schema)}))


if __name__ == "__main__":
    main()
