#!/usr/bin/env python
"""BPP_v2 post-processing: generated episodes to decoded arrays and data labels.

Subcommands (all idempotent; outputs only under --out / --split-dir):
  build  (venv_sim: pyarrow + PIL)  generator split dir (status/, traces/, robomme_data_lerobot/data/) ->
         <split>/robomme_data_lerobot/{meta/*, data/ (hardlinks, byte-identical to the generator's PASS parquets)}
         <split>/cache/imgcache_task0/{images.npy, states.npy, actions.npy, index.json, episode_image_sha256.json,
                                       cache_verification.json}
         <split>/cache/patternlock_easy_password_counter_v2/{manifest.json, episode_XXXXXX.npz}  (schema v2, 18 keys)
         <split>/cache/bpp_v2_events_v1/manifest.json      (logged ACCEPTED events per episode; its sha256 is the
                                                             sidecars' source_memory_manifest_sha256)
         <split>/cache/bpp_v2_row_labels_v1/labels.npz     (row-aligned with the cache: owner, segment, kind, accepted
                                                             event, counter, anchor, target_valid, is_demo)
         <split>/dataset_manifest.json, <split>/reports/build.json
  stats  (any venv with numpy)  D6: coverage of the policy targets under the frozen v1 stats + v2 stats computed the way
         the legacy trainer computed them (`BaseDataset.compute_stats`: openpi RunningStats over every training sample's
         state (1,8) and absolute action chunk (16,8)) -> <split>/cache/<stats dir>/{stats.json, provenance.json,
         coverage.json}; the v1 stats dir is copied alongside.
  yaml   (any venv with numpy + PyYAML)  <split>/SHA256SUMS + the trainer data yaml (schema robomme_data_manifest_v1).

Counters come only from the logged ACCEPTED events (parquet `accepted_event`, cross-checked with the trace's
bpp_v2_summary events), never from pixels. Anchors/targets/mask come from the included
`segment_contract.segment_anchor_expectations` and are cross-checked with an
independent implementation of the T0 §3.4 rule (any mismatch aborts the build).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

H = 16
from segment_contract import segment_anchor_expectations

SEGMENT_CONTRACT_SHA = hashlib.sha256(Path(__file__).with_name("segment_contract.py").read_bytes()).hexdigest()
TREE_LOCK = "e8d12d24586f055ae8f6689259eeb3bc23e651d912e06fc52f010172ad9778cb"
TREE_MANIFEST = "ec5fb900c1b64df57b92b97ec450a677486a77e22bb9654d0ebfd51f61d9d49b"
HOME_QPOS = np.asarray([-0.00147878, 0.11070219, 0.00146709, -2.33247628, -0.00025332, 2.44317490, 0.78557629])
KINDS = ["home", "demo_reach", "return", "handoff_reach", "exec_reach"]
KIND_CODE = {k: i for i, k in enumerate(KINDS)}
IMAGE_KEYS = ("image", "wrist_image")
TEMPLATES = Path(__file__).resolve().parent / "templates"
SCHEMA_NAME = "patternlock_easy_password_counter"


def now():
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_json(path: Path, obj, indent=1, sort_keys=True):
    atomic_bytes(path, (json.dumps(obj, indent=indent, sort_keys=sort_keys) + "\n").encode())


def trainer_segment_anchor_fn():
    """Use the standalone data-label rule included with this bundle."""
    return segment_anchor_expectations


def independent_anchors(owner: np.ndarray, seg: np.ndarray, e0: int):
    """T0 §3.4 rule, written independently: anchor t iff t >= exec_start, t < T-1, owner[t+1] == 1; targets
    min(t+1+j, end) with end = last row of the contiguous policy run containing t+1; mask = t+1+j > end."""
    T = len(owner)
    end = np.full(T, -1, dtype=np.int64)
    for r in range(T - 1, -1, -1):
        if owner[r] == 1:
            end[r] = r if (r == T - 1 or owner[r + 1] != 1 or seg[r + 1] != seg[r]) else end[r + 1]
    valid = np.zeros(T, dtype=bool)
    rows = np.full((T, H), -1, dtype=np.int64)
    pad = np.zeros((T, H), dtype=bool)
    for t in range(e0, T - 1):
        if owner[t + 1] != 1:
            continue
        raw = t + 1 + np.arange(H, dtype=np.int64)
        valid[t] = True
        rows[t] = np.minimum(raw, end[t + 1])
        pad[t] = raw > end[t + 1]
    return valid, rows, pad


# ------------------------------------------------------------------------------------------------------------ build
_SEG_FN = None


def _init_worker():
    global _SEG_FN
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _SEG_FN = trainer_segment_anchor_fn()


def _decode(raw: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _read_parquet(path: Path, with_images: bool):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    T = t.num_rows

    def fixed(col):
        arr = t.column(col).combine_chunks()
        vals = arr.flatten().to_numpy(zero_copy_only=False)
        if vals.dtype != np.float32:
            raise ValueError(f"{path}: {col} dtype {vals.dtype}")
        return vals.reshape(T, 8).copy()

    out = {
        "T": T,
        "state": fixed("state"), "actions": fixed("actions"),
        "owner_str": t.column("control_owner").to_pylist(),
        "segment_id": np.asarray(t.column("segment_id").to_pylist(), dtype=np.int64),
        "kind": t.column("segment_kind").to_pylist(),
        "accepted": np.asarray(t.column("accepted_event").to_pylist(), dtype=np.int64),
        "exec_start_col": np.asarray(t.column("exec_start_idx").to_pylist(), dtype=np.int64),
        "is_demo": np.asarray(t.column("is_demo").to_pylist(), dtype=bool),
        "frame_index": np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64),
        "episode_index": np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64),
        "task_index": np.asarray(t.column("task_index").to_pylist(), dtype=np.int64),
    }
    if with_images:
        img = np.empty((T, 6, 256, 256), np.uint8)
        for k, key in enumerate(IMAGE_KEYS):
            for r, cell in enumerate(t.column(key).to_pylist()):
                a = _decode(cell["bytes"])
                if a.shape != (256, 256, 3):
                    raise ValueError(f"{path}: {key} row {r} shape {a.shape}")
                img[r, 3 * k:3 * k + 3] = a.transpose(2, 0, 1)
        out["images"] = img
    return out


def _episode_job(job):
    """One episode: verify, decode into the shared images memmap, build sidecar + labels. Raises on any violation."""
    ep = int(job["episode"])
    start = int(job["start"])
    st = job["status"]
    pq_path = Path(job["parquet"])

    def fail(msg):
        raise ValueError(f"episode {ep}: {msg}")

    pq_sha = sha256_file(pq_path)
    if pq_sha != st["parquet_sha256"]:
        fail(f"parquet sha {pq_sha} != status {st['parquet_sha256']}")
    d = _read_parquet(pq_path, with_images=True)
    T = d["T"]
    if T != int(st["num_frames"]):
        fail(f"parquet rows {T} != status num_frames {st['num_frames']}")
    fr = np.arange(T)
    if not np.array_equal(d["frame_index"], fr) or np.any(d["episode_index"] != ep) or np.any(d["task_index"] != 0):
        fail("frame_index/episode_index/task_index columns")
    if set(d["owner_str"]) - {"policy", "sim"} or set(d["kind"]) - set(KINDS):
        fail(f"unknown owner/kind values {set(d['owner_str'])} {set(d['kind'])}")
    owner = np.asarray([1 if o == "policy" else 0 for o in d["owner_str"]], dtype=np.int64)
    seg = d["segment_id"]
    kind = d["kind"]
    kcode = np.asarray([KIND_CODE[k] for k in kind], dtype=np.int8)
    acc = d["accepted"]
    e0s = set(d["exec_start_col"].tolist())
    if len(e0s) != 1:
        fail(f"exec_start_idx not constant: {e0s}")
    e0 = int(e0s.pop())
    if e0 != int(st["exec_start_idx"]):
        fail("exec_start_idx != status")
    # is_demo: True for rows < exec_start, False from exec_start on (S1 contract)
    if not np.array_equal(d["is_demo"], fr < e0):
        fail("is_demo does not switch exactly at exec_start_idx")
    # ownership/segment conventions (S1): policy on demo_reach/exec_reach, sim elsewhere; segment -1 exactly on sim
    pol_kind = np.isin(kcode, [KIND_CODE["demo_reach"], KIND_CODE["exec_reach"]])
    if not np.array_equal(owner == 1, pol_kind):
        fail("control_owner disagrees with segment_kind")
    if np.any(seg[owner == 0] != -1) or np.any(seg[owner == 1] < 0):
        fail("segment_id not -1 exactly on sim rows")
    if kind[0] != "home" or owner[0] != 0 or kind[-1] != "home" or owner[-1] != 0:
        fail("row 0 / last row are not sim-owned home rows")
    if kind[e0] != "home" or owner[e0] != 0 or e0 + 1 >= T or kind[e0 + 1] != "exec_reach" or owner[e0 + 1] != 1:
        fail("exec_start is not a settled home followed by an exec_reach")
    if not np.allclose(d["state"][0, :7], HOME_QPOS, atol=1e-7) or d["state"][0, 7] != 0.0:
        fail("row-0 state is not HOME")
    if np.any(d["actions"][:, 7] != -1.0) or np.any(d["state"][:, 7] != 0.0):
        fail("state[:,7] != 0 or actions[:,7] != -1")
    # ---- logged ACCEPTED events: parquet column vs trace summary (never pixels)
    pq_events = [(int(r), int(acc[r]), kind[r]) for r in np.flatnonzero(acc >= 0)]
    trace = json.loads(gzip.decompress(Path(job["trace"]).read_bytes()))
    if sha256_file(job["trace"]) != st["trace_sha256"]:
        fail("trace sha != status")
    summ = trace["bpp_v2_summary"]
    tr_events = [(int(e["row"]), int(e["button"]), str(e["reach_kind"])) for e in summ["events"]]
    if pq_events != tr_events:
        fail(f"accepted_event column {pq_events} != trace events {tr_events}")
    if not summ.get("success") or summ.get("failed") or int(summ.get("rows", -1)) != T:
        fail("trace summary not success / rows mismatch")
    pw = [int(x) for x in st["full_password"]]
    exec_pw = pw[1:]
    L = len(exec_pw)
    demo = [b for r, b, k in pq_events if k == "demo_reach"]
    hand = [b for r, b, k in pq_events if k == "handoff_reach"]
    exe = [(r, b) for r, b, k in pq_events if k == "exec_reach"]
    if demo != pw or hand != [pw[0]] or [b for _, b in exe] != exec_pw:
        fail(f"event buttons demo {demo} handoff {hand} exec {exe} vs password {pw}")
    if [k for _, _, k in pq_events] != ["demo_reach"] * len(pw) + ["handoff_reach"] + ["exec_reach"] * L:
        fail("event order")
    if not all(r < e0 for r, _, k in pq_events if k != "exec_reach") or not all(r > e0 for r, _ in exe):
        fail("events on the wrong side of exec_start")
    trans = np.zeros(T, dtype=bool)
    for c, _ in exe:
        # boundary-row rule: the contact row keeps its (policy) action; the return starts on the next (sim) row
        if owner[c] != 1 or kind[c] != "exec_reach" or c + 1 >= T or kind[c + 1] != "return" or owner[c + 1] != 0:
            fail(f"exec contact row {c} does not hand over to a sim return row")
        trans[c + 1] = True
    counter = np.where(fr >= e0, np.cumsum(trans), -1).astype(np.int64)
    if counter[e0] != 0 or counter[-1] != L:
        fail("counter endpoints")
    # ---- anchors / targets / mask: trainer reference builder + independent implementation
    v, rows, pad = _SEG_FN({"control_owner": owner, "segment_id": seg}, episode=ep, num_frames=T, exec_start=e0,
                           transition=trans)
    v2, rows2, pad2 = independent_anchors(owner, seg, e0)
    mism = int((v != v2).sum() + (rows != rows2).any(axis=1).sum() + (pad != pad2).any(axis=1).sum())
    if mism:
        fail(f"{mism} anchor/target mismatches between trainer builder and independent rule")
    anchors = np.flatnonzero(v)
    n_policy_exec = int(owner[e0 + 1:].sum())
    if len(anchors) != n_policy_exec:
        fail(f"anchors {len(anchors)} != policy-owned exec rows {n_policy_exec}")
    tr = rows[anchors]
    if np.any(owner[tr] != 1) or np.any(seg[tr] != seg[anchors + 1][:, None]):
        fail("a target row is not policy-owned in the anchor's segment")
    if np.any(counter[anchors] != counter[anchors + 1]):
        fail("counter changes inside an anchor's decision (counter[t] != counter[t+1])")
    if np.any(counter[anchors] >= L):
        fail("anchor after the final press")
    home_anchors = [int(t) for t in anchors if kind[t] == "home"]
    arrays = dict(
        schema_name=np.asarray(SCHEMA_NAME), schema_version=np.asarray(2, dtype=np.int64),
        episode_index=np.asarray(ep, dtype=np.int64), source_sha256=np.asarray(pq_sha),
        source_num_frames=np.asarray(T, dtype=np.int64), exec_start_idx=np.asarray(e0, dtype=np.int64),
        execution_password_ids=np.asarray(exec_pw, dtype=np.int64),
        execution_password_length=np.asarray(L, dtype=np.int64),
        frame_index=np.arange(T, dtype=np.int64), counter=counter, counter_valid=fr >= e0,
        counter_transition=trans, action_pred_horizon=np.asarray(H, dtype=np.int64),
        control_owner=owner, segment_id=seg, sample_valid=v, action_target_rows=rows, action_padding_mask=pad)
    bio = io.BytesIO()
    np.savez(bio, **arrays)
    side_bytes = bio.getvalue()
    side_path = Path(job["sidecar_dir"]) / f"episode_{ep:06d}.npz"
    atomic_bytes(side_path, side_bytes)
    # ---- images into the shared memmap (disjoint row range per episode)
    mm = np.load(job["images_path"], mmap_mode="r+")
    mm[start:start + T] = d["images"]
    mm.flush()
    del mm
    target_valid = (owner == 1) & (fr > e0)
    return {
        "episode": ep, "start": start, "T": T, "exec_start": e0, "L": L, "parquet_sha256": pq_sha,
        "image_sha256": sha256_bytes(d["images"].tobytes()),
        "sidecar_sha256": sha256_bytes(side_bytes), "sidecar_file": side_path.name,
        "anchors": int(len(anchors)), "padded_anchors": int(pad[anchors].any(axis=1).sum()),
        "padded_targets": int(pad[anchors].sum()), "home_anchors": home_anchors,
        "counter_rise_rows": [int(r) for r in np.flatnonzero(trans)],
        "anchor_counter_hist": np.bincount(counter[anchors], minlength=4).tolist(),
        "events": [list(e) for e in pq_events],
        "rows_by_kind": {k: int((kcode == i).sum()) for i, k in enumerate(KINDS)},
        "rows_by_owner": {"policy": int(owner.sum()), "sim": int(T - owner.sum())},
        "state": d["state"], "actions": d["actions"],
        "labels": {"control_owner": owner.astype(np.int8), "segment_id": seg.astype(np.int32),
                   "segment_kind": kcode, "accepted_event": acc.astype(np.int8), "counter": counter.astype(np.int8),
                   "anchor": v, "target_valid": target_valid, "is_demo": d["is_demo"]},
    }


def _verify_job(job):
    """Second pass: re-decode the published (hardlinked) parquet and compare every row with the cache."""
    ep, start, T = int(job["episode"]), int(job["start"]), int(job["T"])
    d = _read_parquet(Path(job["parquet"]), with_images=True)
    imgs = np.load(job["images_path"], mmap_mode="r")[start:start + T]
    st = np.load(job["states_path"], mmap_mode="r")[start:start + T]
    ac = np.load(job["actions_path"], mmap_mode="r")[start:start + T]
    return {"episode": ep, "rows_equal": d["T"] == T,
            "images_equal": bool(d["T"] == T and np.array_equal(d["images"], imgs)),
            "states_equal": bool(d["T"] == T and np.array_equal(d["state"], st)),
            "actions_equal": bool(d["T"] == T and np.array_equal(d["actions"], ac)),
            "image_sha256": sha256_bytes(np.ascontiguousarray(imgs).tobytes())}


def load_selection(path: Path) -> dict[int, dict]:
    m = json.loads(path.read_text())
    recs = m.get("records") or m.get("episodes")
    out = {int(r["episode"]): r for r in recs}
    if len(out) != len(recs):
        raise SystemExit("selection has duplicate episodes")
    return out


def cmd_build(a):
    t_start = time.time()
    src, out = Path(a.src), Path(a.out)
    sel = load_selection(Path(a.selection))
    sel_sha = sha256_file(a.selection)
    status_files = sorted((src / "status").glob("episode_*.json"))
    statuses = {}
    for p in status_files:
        st = json.loads(p.read_text())
        e = int(st["episode"])
        if e not in sel:
            continue  # e.g. pilot dirs hold several splits' ids under one selection? (never for gen/<split>)
        statuses[e] = (st, p)
    eps = sorted(statuses)
    if a.episodes:
        want = sorted(int(x) for x in a.episodes.split(","))
        eps = [e for e in eps if e in set(want)]
    if not a.subset and set(eps) != set(sel):
        missing = sorted(set(sel) - set(eps))
        raise SystemExit(f"{len(missing)} selection episodes have no PASS status (first {missing[:10]}); "
                         f"use --subset only for pilot/dev builds")
    if not eps:
        raise SystemExit("no episodes")
    problems = []
    for e in eps:
        st, p = statuses[e]
        r = sel[e]
        if st.get("status") != "PASS" or st.get("schema") != "bpp_v2_generation_status_v1":
            problems.append(f"{e}: status {st.get('status')}")
        if st["tree"]["lock_sha256"] != TREE_LOCK or st["tree"]["tree_manifest_sha256"] != TREE_MANIFEST:
            problems.append(f"{e}: sim tree {st['tree']}")
        if int(st["seed"]) != int(r["seed"]) or [int(x) for x in st["full_password"]] != [int(x) for x in r["full_password"]]:
            problems.append(f"{e}: identity drift vs selection")
        if st.get("source") != r.get("source"):
            problems.append(f"{e}: source {st.get('source')} != selection {r.get('source')}")
        if st.get("split") != a.split_kind:
            problems.append(f"{e}: status split {st.get('split')} != {a.split_kind}")
        if [int(x) for x in st["execution_password_ids"]] != [int(x) for x in st["full_password"]][1:]:
            problems.append(f"{e}: execution_password_ids")
        al = st["gates"].get("alignment", {})
        if not (st["gates"].get("events_ok") and st["gates"].get("structure_ok") and st["gates"]["contract"].get("ok")
                and st["gates"]["home"].get("ok") and st["gates"]["contacts"].get("ok") and al.get("ok")
                and st["gates"].get("red_pixels") == 0 and st["gates"].get("parquet_equals_live")):
            problems.append(f"{e}: a generation gate is not ok")
    if problems:
        raise SystemExit("status problems:\n" + "\n".join(problems[:50]))
    log(f"build split={a.split} src={src} episodes={len(eps)} (selection {len(sel)})")

    # --- output tree (fresh)
    if out.exists():
        if not a.force:
            raise SystemExit(f"{out} exists (use --force to rebuild)")
        for sub in ("cache", "robomme_data_lerobot", "reports", "SHA256SUMS", "dataset_manifest.json"):
            p = out / sub
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    lr = out / "robomme_data_lerobot"
    cache = out / "cache/imgcache_task0"
    side = out / "cache/patternlock_easy_password_counter_v2"
    evdir = out / "cache/bpp_v2_events_v1"
    labdir = out / "cache/bpp_v2_row_labels_v1"
    for p in (lr / "meta", lr / "data", cache, side, evdir, labdir, out / "reports"):
        p.mkdir(parents=True, exist_ok=True)

    # --- parquets: hardlink (same /scratch fs; byte-identical, immutable publish of the generator's file)
    rel_pq = {}
    linked = copied = 0
    for e in eps:
        st, _ = statuses[e]
        srcp = Path(st["parquet"])
        if not srcp.is_file():  # status written on the node that generated it; fall back to the src dir layout
            srcp = src / "robomme_data_lerobot/data" / f"chunk-{e // 1000:03d}" / f"episode_{e:06d}.parquet"
        rel = f"data/chunk-{e // 1000:03d}/episode_{e:06d}.parquet"
        dst = lr / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(srcp, dst)
            linked += 1
        except OSError:
            shutil.copy2(srcp, dst)
            copied += 1
        rel_pq[e] = rel
    log(f"parquets: {linked} hardlinked, {copied} copied")

    # --- row layout (episode-contiguous, ascending episode index)
    starts, s = {}, 0
    for e in eps:
        starts[e] = s
        s += int(statuses[e][0]["num_frames"])
    N = s
    images_path = cache / "images.npy"
    mm = np.lib.format.open_memmap(images_path, mode="w+", dtype=np.uint8, shape=(N, 6, 256, 256))
    del mm
    log(f"frames N={N}; images.npy {N * 6 * 256 * 256 / 1e9:.1f} GB (sparse, filled in parallel)")
    jobs = [{"episode": e, "start": starts[e], "status": statuses[e][0], "parquet": str(lr / rel_pq[e]),
             "trace": statuses[e][0]["trace"] if Path(statuses[e][0]["trace"]).is_file()
             else str(src / "traces" / f"episode_{e:06d}.json.gz"),
             "sidecar_dir": str(side), "images_path": str(images_path)} for e in eps]
    results = []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.map(_episode_job, jobs, chunksize=1)):
            results.append(r)
            if (i + 1) % 250 == 0 or i + 1 == len(jobs):
                log(f"[build] {i + 1}/{len(jobs)} episodes")
    assert [r["episode"] for r in results] == eps

    states = np.concatenate([r["state"] for r in results]).astype(np.float32)
    actions = np.concatenate([r["actions"] for r in results]).astype(np.float32)
    assert states.shape == (N, 8) and actions.shape == (N, 8)
    np.save(cache / "states.npy", states)
    np.save(cache / "actions.npy", actions)
    index = {"num_frames": N,
             "episodes": [{"episode_index": r["episode"], "start": r["start"], "length": r["T"],
                           "exec_start": r["exec_start"], "task_index": 0} for r in results],
             "image_keys": list(IMAGE_KEYS), "task_indices": [0], "visual_variant": "bpp_v2",
             "dataset_root": str(lr), "row_convention": "row r = env step r (row 0 = home reset frame); "
             "actions[r] = applied pd_joint_pos command that produced states[r]/images[r]"}
    atomic_json(cache / "index.json", index)
    atomic_json(cache / "episode_image_sha256.json", {str(r["episode"]): r["image_sha256"] for r in results},
                indent=0)

    # --- row labels (aligned with cache rows)
    lab = {k: np.concatenate([r["labels"][k] for r in results]) for k in results[0]["labels"]}
    lab["episode_index"] = np.concatenate([np.full(r["T"], r["episode"], dtype=np.int32) for r in results])
    lab["frame_index"] = np.concatenate([np.arange(r["T"], dtype=np.int32) for r in results])
    lab["segment_kind_names"] = np.asarray(KINDS)
    bio = io.BytesIO()
    np.savez(bio, **lab)
    atomic_bytes(labdir / "labels.npz", bio.getvalue())

    # --- events manifest (identity of the counter source) -> sidecar manifest
    ev_records = []
    for r in results:
        st, sp = statuses[r["episode"]]
        ev_records.append({"episode_index": r["episode"], "num_frames": r["T"], "exec_start_idx": r["exec_start"],
                           "full_password": [int(x) for x in st["full_password"]],
                           "accepted_events": r["events"], "counter_rise_rows": r["counter_rise_rows"],
                           "parquet_sha256": r["parquet_sha256"], "trace_sha256": st["trace_sha256"],
                           "status_sha256": sha256_file(sp)})
    events_manifest = {
        "schema": "bpp_v2_accepted_events_v1", "split": a.split, "num_episodes": len(ev_records),
        "source": "parquet column accepted_event (logged by the v2 env inside evaluate()), cross-checked equal to "
                  "the trace bpp_v2_summary.events; never pixels",
        "counter_rule": "counter[t] = #accepted exec_reach presses detected at rows < t (v1 logical+1); rise at "
                        "contact row + 1 = first (sim-owned) return row; -1 before exec_start_idx",
        "episodes": ev_records}
    atomic_json(evdir / "manifest.json", events_manifest)
    events_sha = sha256_file(evdir / "manifest.json")

    v1_splits = {}
    if a.v1_splits:
        vm = json.loads(Path(a.v1_splits).read_text())
        v1_splits = {k: [int(x) for x in v] for k, v in vm["splits"].items()}
    ids = [r["episode"] for r in results]
    splits = {"train_all": ids}
    for k in ("dev_train", "dev_val"):
        if k in v1_splits:
            splits[k] = [e for e in v1_splits[k] if e in set(ids)]
    if not a.subset and v1_splits and set(v1_splits.get("train_all", [])) != set(ids):
        raise SystemExit("v1 train_all != this split's episodes")
    n_samples = sum(r["anchors"] for r in results)
    cdist = np.sum([r["anchor_counter_hist"] for r in results], axis=0).tolist()
    counter_manifest = {
        "schema_name": SCHEMA_NAME, "schema_version": 2, "num_episodes": len(results),
        "episodes": [{"episode_index": r["episode"], "source_sha256": r["parquet_sha256"], "num_frames": r["T"],
                      "file": r["sidecar_file"], "sidecar_sha256": r["sidecar_sha256"], "num_samples": r["anchors"],
                      "exec_start_idx": r["exec_start"], "execution_password_length": r["L"],
                      "num_counter_transitions": r["L"], "padded_anchors": r["padded_anchors"],
                      "home_anchors": r["home_anchors"],
                      "source_file": str(lr / rel_pq[r["episode"]])} for r in results],
        "splits": splits,
        "contract": {
            "conditioning_allowlist": ["execution_password_ids", "counter"],
            "anchor_rule": "obs row t is a sample iff t >= exec_start_idx and action row t+1 is policy-owned "
                           "(standalone segment_contract.segment_anchor_expectations)",
            "causal_action_alignment": "observation row t predicts action rows t+1..t+16 of the same policy segment",
            "terminal_action_padding": "targets past the segment end repeat the last in-segment action; mask stored "
                                       "(action_padding_mask); loss unchanged in form (v1 terminal clamp)",
            "counter_alignment": "logical+1 from logged ACCEPTED execution events (never pixels)",
            "a2_oracle": "next target = execution_password_ids[counter[t+1]] (logical); counter[t+1] == counter[t] "
                         "at every anchor",
            "recorder_alignment": "source row t stores the applied action that produced observation t",
            "sim_rows": "row 0, sim pw[0] handoff reach, every return row and every settled-home row are sim-owned: "
                        "history only, never targets",
            "forbidden_conditioning": ["next-button ID", "next-button coordinates", "logical press frame",
                                       "future counters", "control_owner/segment metadata"],
        },
        "num_samples": n_samples, "counter_distribution": {str(i): int(c) for i, c in enumerate(cdist)},
        "source_memory_manifest_sha256": events_sha,
        "source_memory_manifest": "cache/bpp_v2_events_v1/manifest.json",
        "source_dataset_root": str(lr),
        "visual_variant": {"name": "bpp_v2", "description": "BPP_v1 visuals (no flash, no trail) + automatic "
                           "return-home after every accepted press (20-step joint interpolation + snap)"},
        "segment_builder": {"file": "generation/segment_contract.py", "sha256": SEGMENT_CONTRACT_SHA,
                            "function": "segment_anchor_expectations"},
    }
    atomic_json(side / "manifest.json", counter_manifest)

    # --- LeRobot meta
    tmpl = json.loads((TEMPLATES / "info.json").read_text())
    info = dict(tmpl)
    feats = dict(tmpl["features"])
    feats["control_owner"] = {"dtype": "string", "names": ["policy (policy/expert-owned action) or sim"],
                              "shape": [1]}
    feats["segment_id"] = {"dtype": "int32", "names": ["policy reach segment id; -1 on sim rows"], "shape": [1]}
    feats["segment_kind"] = {"dtype": "string", "names": ["home|demo_reach|return|handoff_reach|exec_reach"],
                             "shape": [1]}
    feats["accepted_event"] = {"dtype": "int32", "names": ["button accepted at this row, else -1"], "shape": [1]}
    info["features"] = feats
    info.update({"total_episodes": len(results), "total_frames": N,
                 "total_chunks": len({e // 1000 for e in ids}), "splits": {"train": f"0:{len(results)}"},
                 "chunks_size": 1000,
                 "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"})
    atomic_json(lr / "meta/info.json", info, indent=2)
    task = json.loads((TEMPLATES / "tasks.jsonl").read_text().splitlines()[0])["task"]
    atomic_bytes(lr / "meta/tasks.jsonl", (TEMPLATES / "tasks.jsonl").read_bytes())
    atomic_bytes(lr / "meta/episodes.jsonl", "".join(
        json.dumps({"episode_index": r["episode"], "tasks": [task], "length": r["T"]}) + "\n"
        for r in results).encode())

    # --- second pass: cache == parquets (every row, both cameras, states, actions)
    log("verify pass: re-decoding every parquet")
    vjobs = [{"episode": r["episode"], "start": r["start"], "T": r["T"], "parquet": str(lr / rel_pq[r["episode"]]),
              "images_path": str(images_path), "states_path": str(cache / "states.npy"),
              "actions_path": str(cache / "actions.npy")} for r in results]
    with ProcessPoolExecutor(a.workers) as pool:
        ver = list(pool.map(_verify_job, vjobs, chunksize=2))
    bad = [v["episode"] for v in ver if not (v["rows_equal"] and v["images_equal"] and v["states_equal"]
                                             and v["actions_equal"])]
    sha_bad = [v["episode"] for v, r in zip(ver, results) if v["image_sha256"] != r["image_sha256"]]
    cver = {"frames": N, "episodes": len(results), "mismatched_episodes": bad, "image_sha_mismatch": sha_bad,
            "images_shape_ok": bool(np.load(images_path, mmap_mode="r").shape == (N, 6, 256, 256)),
            "images_bytes": images_path.stat().st_size, "states_bytes": (cache / "states.npy").stat().st_size,
            "actions_bytes": (cache / "actions.npy").stat().st_size,
            "status": "PASS" if not bad and not sha_bad else "FAIL", "at": now()}
    atomic_json(cache / "cache_verification.json", cver)
    if cver["status"] != "PASS":
        raise SystemExit(f"cache verification FAIL: {bad[:10]} {sha_bad[:10]}")

    # --- dataset manifest
    recs = []
    for r in results:
        st, sp = statuses[r["episode"]]
        recs.append({"episode_index": r["episode"], "seed": int(st["seed"]), "full_password": st["full_password"],
                     "source": st.get("source"), "variant_index": st.get("variant_index"),
                     "num_frames": r["T"], "exec_start_idx": r["exec_start"], "parquet": rel_pq[r["episode"]],
                     "parquet_sha256": r["parquet_sha256"], "status_sha256": sha256_file(sp),
                     "trace_sha256": st["trace_sha256"], "rowlog_sha256": st["rowlog_sha256"],
                     "rows_by_kind": r["rows_by_kind"], "rows_by_owner": r["rows_by_owner"],
                     "anchors": r["anchors"], "gen": {"run": st.get("run"), "host": st.get("host"),
                                                      "attempt": st.get("attempt"),
                                                      "worker_sha256": st.get("worker_sha256"),
                                                      "alignment_policy_max_abs_diff":
                                                          st["gates"]["alignment"].get("policy_max_abs_diff")}})
    dm = {"artifact_id": f"dataset:patternlock_easy_bpp_v2_{a.split}", "visual_variant": "bpp_v2",
          "num_episodes": len(recs), "num_frames": N, "num_anchors": n_samples,
          "selection_manifest": str(a.selection), "selection_manifest_sha256": sel_sha,
          "sim_tree": {"lock_sha256": TREE_LOCK, "tree_manifest_sha256": TREE_MANIFEST},
          "generator_src_dir": str(src), "episodes": recs, "status": "PASS", "built_at": now()}
    atomic_json(out / "dataset_manifest.json", dm)

    rep = {"split": a.split, "episodes": len(results), "frames": N, "anchors": n_samples,
           "padded_anchors": sum(r["padded_anchors"] for r in results),
           "padded_targets": sum(r["padded_targets"] for r in results),
           "home_anchors": sum(len(r["home_anchors"]) for r in results),
           "anchor_counter_hist": cdist,
           "rows_by_kind": {k: sum(r["rows_by_kind"][k] for r in results) for k in KINDS},
           "rows_by_owner": {k: sum(r["rows_by_owner"][k] for r in results) for k in ("policy", "sim")},
           "parquets_hardlinked": linked, "parquets_copied": copied,
           "parquet_bytes": sum((lr / rel_pq[e]).stat().st_size for e in eps),
           "cache_bytes": {k: (cache / k).stat().st_size for k in ("images.npy", "states.npy", "actions.npy")},
           "anchor_mismatches_trainer_vs_independent": 0, "cache_verification": cver["status"],
           "events_manifest_sha256": events_sha, "counter_manifest_sha256": sha256_file(side / "manifest.json"),
           "dataset_manifest_sha256": sha256_file(out / "dataset_manifest.json"),
           "index_sha256": sha256_file(cache / "index.json"), "info_sha256": sha256_file(lr / "meta/info.json"),
           "seconds": round(time.time() - t_start, 1), "built_at": now(), "p5_build_sha256": sha256_file(__file__)}
    atomic_json(out / "reports/build.json", rep)
    log("BUILD PASS " + json.dumps({k: rep[k] for k in ("split", "episodes", "frames", "anchors", "seconds")}))


# ------------------------------------------------------------------------------------------------------------ stats
class RunningStats:
    """Verbatim copy of the legacy trainer's openpi `RunningStats` (memory_diffusion_policy/eval_envs/utils/
    normalize.py, BPP_v1 code_post) — the code that produced the frozen v1 stats via BaseDataset.compute_stats."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000

    def update(self, batch: np.ndarray) -> None:
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()
        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)
        self._update_histograms(batch)

    def get_statistics(self) -> dict:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        median = self._compute_quantiles([0.5])[0]
        return {"mean": self._mean, "std": stddev, "min": self._min, "max": self._max, "q01": q01, "q99": q99,
                "median": median}

    def _adjust_histograms(self):
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def minmax(x, st):
    lo, hi = np.asarray(st["min"]), np.asarray(st["max"])
    return ((x - lo) / (hi - lo + 1e-6) * 2.0 - 1.0).astype(np.float32)  # == src.data_il.minmax_normalize


def split_samples(split_dir: Path):
    """Yield (episode, anchor_states (n,8), target_chunks (n,16,8), unique target rows (m,8)) in trainer order."""
    cache = split_dir / "cache/imgcache_task0"
    side = split_dir / "cache/patternlock_easy_password_counter_v2"
    index = json.loads((cache / "index.json").read_text())
    states = np.load(cache / "states.npy")
    actions = np.load(cache / "actions.npy")
    man = json.loads((side / "manifest.json").read_text())
    files = {int(r["episode_index"]): r["file"] for r in man["episodes"]}
    for rec in sorted(index["episodes"], key=lambda r: int(r["episode_index"])):
        e, s = int(rec["episode_index"]), int(rec["start"])
        with np.load(side / files[e]) as z:
            v = np.asarray(z["sample_valid"], dtype=bool)
            rows = np.asarray(z["action_target_rows"], dtype=np.int64)
        anc = np.flatnonzero(v)
        uniq = np.unique(rows[anc])
        yield e, states[s + anc], actions[s + rows[anc]], actions[s + uniq]


def coverage(split_dir: Path, st: dict) -> dict:
    out = {"target_chunk_values": 0, "target_chunk_values_outside_pm1": 0, "unique_target_rows": 0,
           "unique_target_rows_outside_pm1": 0, "anchor_states": 0, "anchor_state_values_outside_pm1": 0,
           "episodes_with_target_outside": 0}
    jmin_t, jmax_t = np.full(7, np.inf), np.full(7, -np.inf)
    jmin_s, jmax_s = np.full(7, np.inf), np.full(7, -np.inf)
    out_joint = np.zeros(7, dtype=np.int64)
    for e, S, A, U in split_samples(split_dir):
        na = minmax(A.reshape(-1, 8), st["action"])[:, :7]
        nu = minmax(U, st["action"])[:, :7]
        ns = minmax(S, st["state"])[:, :7]
        out["target_chunk_values"] += na.size
        o = np.abs(na) > 1.0
        out["target_chunk_values_outside_pm1"] += int(o.sum())
        out["unique_target_rows"] += len(nu)
        ou = (np.abs(nu) > 1.0)
        out["unique_target_rows_outside_pm1"] += int(ou.any(axis=1).sum())
        out_joint += ou.sum(axis=0)
        out["episodes_with_target_outside"] += int(o.any())
        out["anchor_states"] += len(ns)
        out["anchor_state_values_outside_pm1"] += int((np.abs(ns) > 1.0).sum())
        jmin_t, jmax_t = np.minimum(jmin_t, na.min(0)), np.maximum(jmax_t, na.max(0))
        jmin_s, jmax_s = np.minimum(jmin_s, ns.min(0)), np.maximum(jmax_s, ns.max(0))
    out.update({"target_per_joint_min": [round(float(x), 4) for x in jmin_t],
                "target_per_joint_max": [round(float(x), 4) for x in jmax_t],
                "unique_target_rows_outside_per_joint": out_joint.tolist(),
                "anchor_state_per_joint_min": [round(float(x), 4) for x in jmin_s],
                "anchor_state_per_joint_max": [round(float(x), 4) for x in jmax_s]})
    out["all_targets_inside_pm1"] = out["target_chunk_values_outside_pm1"] == 0
    out["all_anchor_states_inside_pm1"] = out["anchor_state_values_outside_pm1"] == 0
    out["unique_target_rows_outside_fraction"] = round(out["unique_target_rows_outside_pm1"]
                                                       / max(1, out["unique_target_rows"]), 5)
    return out


def load_stats_json(p: Path) -> dict:
    d = json.loads(p.read_text())
    d = d.get("norm_stats", d)
    return {k: {kk: np.asarray(vv) for kk, vv in d[k].items()} for k in ("state", "action")}


def cmd_stats(a):
    split_dir = Path(a.split_dir)
    v1p = Path(a.v1_stats) if a.v1_stats else None
    v1_sha, v1, cov_v1, v1dir = None, None, None, None
    if v1p is not None:
        v1_sha = sha256_file(v1p)
        if a.v1_sha and v1_sha != a.v1_sha:
            raise SystemExit(f"v1 stats sha {v1_sha} != {a.v1_sha}")
        v1 = load_stats_json(v1p)
        cov_v1 = coverage(split_dir, v1)
        log("coverage under v1 stats: " + json.dumps({k: cov_v1[k] for k in (
            "target_chunk_values_outside_pm1", "unique_target_rows_outside_pm1", "unique_target_rows",
            "target_per_joint_min", "target_per_joint_max", "anchor_state_values_outside_pm1")}))
    # v2 stats, legacy compute_stats semantics: every training sample contributes its state (1,8) and its abs action
    # chunk (16,8); batched per episode in dataset order (min/max exact; mean/std/quantiles informational)
    rs = {"state": RunningStats(), "action": RunningStats()}
    n = 0
    for e, S, A, U in split_samples(split_dir):
        rs["state"].update(S.reshape(-1, 8))
        rs["action"].update(A.reshape(-1, 8))
        n += len(S)
    stats = {k: {kk: [float(x) for x in np.asarray(vv).reshape(-1)] for kk, vv in r.get_statistics().items()}
             for k, r in rs.items()}
    payload = {"norm_stats": stats}
    v2dir = split_dir / "cache" / a.v2_name
    v2dir.mkdir(parents=True, exist_ok=True)
    atomic_bytes(v2dir / "stats.json", json.dumps(payload, indent=2).encode())
    v2 = load_stats_json(v2dir / "stats.json")
    cov_v2 = coverage(split_dir, v2)
    v2_sha = sha256_file(v2dir / "stats.json")
    prov = {"contract": "BPP_v2 policy-target normalization statistics (D6): legacy BaseDataset.compute_stats "
                        "semantics (openpi RunningStats over every training sample: anchor state (1,8) + absolute "
                        "action chunk (16,8) incl. repeat-last targets); only min/max are used by the trainer",
            "computed_on": str(split_dir), "samples": n, "stats_sha256": v2_sha,
            "counter_manifest_sha256": sha256_file(split_dir / "cache/patternlock_easy_password_counter_v2/manifest.json"),
            "index_sha256": sha256_file(split_dir / "cache/imgcache_task0/index.json"),
            "batching": "RunningStats.update once per episode (all its samples) in ascending episode order",
            "computed_at": now(), "p5_build_sha256": sha256_file(__file__)}
    atomic_json(v2dir / "provenance.json", prov)
    if v1p is not None:
        v1dir = split_dir / "cache" / v1p.parent.name
        if v1dir.resolve() != v1p.parent.resolve():
            v1dir.mkdir(parents=True, exist_ok=True)
            for f in v1p.parent.iterdir():
                if f.is_file():
                    shutil.copy2(f, v1dir / f.name)
            if sha256_file(v1dir / "stats.json") != v1_sha:
                raise SystemExit("v1 stats copy differs")
    decision = "v1" if cov_v1 is not None and cov_v1["all_targets_inside_pm1"] else "v2"
    cov = {"d6_rule": "Use supplied v1 stats only when all targets fit [-1,1]; otherwise use v2 stats",
           "decision": decision,
           "selected_stats": str((v1dir if decision == "v1" else v2dir).relative_to(split_dir) / "stats.json"),
           "selected_sha256": v1_sha if decision == "v1" else v2_sha,
           "v1": ({"path": str(v1dir.relative_to(split_dir) / "stats.json"), "sha256": v1_sha, "coverage": cov_v1}
                  if v1dir is not None else None),
           "v2": {"path": str(v2dir.relative_to(split_dir) / "stats.json"), "sha256": v2_sha, "coverage": cov_v2,
                  "min": {k: [round(x, 6) for x in stats[k]["min"]] for k in stats},
                  "max": {k: [round(x, 6) for x in stats[k]["max"]] for k in stats}},
           "v1_minmax": ({k: {"min": [round(float(x), 6) for x in v1[k]["min"]],
                                 "max": [round(float(x), 6) for x in v1[k]["max"]]} for k in v1}
                         if v1 is not None else None),
           "samples": n, "at": now()}
    atomic_json(v2dir / "coverage.json", cov)
    atomic_json(split_dir / "reports/d6_coverage.json", cov)
    log(f"D6 decision={decision} v2 sha={v2_sha} samples={n}")
    print(json.dumps({"decision": decision, "v1_outside": cov_v1["target_chunk_values_outside_pm1"] if cov_v1 else None,
                      "v1_unique_rows_outside": cov_v1["unique_target_rows_outside_pm1"] if cov_v1 else None,
                      "v2_outside": cov_v2["target_chunk_values_outside_pm1"], "v2_sha256": v2_sha}))


def cmd_coverage(a):
    """Coverage of one split's targets under a given stats file (e.g. dev48 under the train v2 stats)."""
    st = load_stats_json(Path(a.stats))
    cov = coverage(Path(a.split_dir), st)
    cov["stats"] = str(a.stats)
    cov["stats_sha256"] = sha256_file(a.stats)
    atomic_json(Path(a.out), cov)
    print(json.dumps(cov))


# ------------------------------------------------------------------------------------------------------------ yaml
def write_sums(split_dir: Path):
    files = []
    for dp, dns, fns in os.walk(split_dir):
        dns.sort()
        rel_dir = Path(dp).relative_to(split_dir)
        if rel_dir.parts[:2] == ("robomme_data_lerobot", "data") or rel_dir.parts[:1] == ("reports",):
            dns[:] = []
            continue
        for f in sorted(fns):
            rel = rel_dir / f
            if str(rel) in ("SHA256SUMS", "STAGED.json") or f == "images.npy" or f.startswith("."):
                continue
            files.append(rel)
    with ProcessPoolExecutor(16) as pool:
        hashes = list(pool.map(sha256_file, [split_dir / f for f in files], chunksize=32))
    atomic_bytes(split_dir / "SHA256SUMS", "".join(f"{h}  {f}\n" for h, f in zip(hashes, files)).encode())
    return len(files)


def cmd_yaml(a):
    import yaml
    root = Path(a.root)
    split_dir = root / a.split
    lr = split_dir / "robomme_data_lerobot"
    cache = split_dir / "cache/imgcache_task0"
    side = split_dir / "cache/patternlock_easy_password_counter_v2"
    stats_path = split_dir / a.stats_rel
    if a.copy_stats_from:  # e.g. dev48 carries a byte-identical copy of the train stats dir
        srcd = Path(a.copy_stats_from)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        for f in srcd.iterdir():
            if f.is_file() and f.name in ("stats.json", "provenance.json", "coverage.json"):
                shutil.copy2(f, stats_path.parent / f.name)
    if a.expect_stats_sha and sha256_file(stats_path) != a.expect_stats_sha:
        raise SystemExit("stats sha differs from --expect-stats-sha")
    nsums = write_sums(split_dir)
    info = json.loads((lr / "meta/info.json").read_text())
    cm = json.loads((side / "manifest.json").read_text())
    dm_sha = sha256_file(split_dir / "dataset_manifest.json")
    rel = lambda p: str(Path(p).relative_to(root))  # noqa: E731
    dev_sel = Path(a.dev_selection_dir)
    f100 = Path(a.fresh100_metadata)
    f100_recs = json.loads(f100.read_text())["records"]
    manifest = {
        "schema": "robomme_data_manifest_v1",
        "artifact_id": f"patternlock_easy_bpp_v2_{a.split}",
        "readiness": a.readiness,
        "path_base": "robomme_root",
        "large_artifact_verification": "small_identity_hashes_and_array_sizes",
        "stages": {
            "lerobot": {
                "path": rel(lr),
                "metadata": {"path": "meta/info.json", "sha256": sha256_file(lr / "meta/info.json")},
                "dataset_manifest_sha256": dm_sha,
                "data_path": info["data_path"], "chunks_size": int(info["chunks_size"]),
                "episodes": int(info["total_episodes"]), "frames": int(info["total_frames"]),
            },
            "decoded_cache": {
                "path": rel(cache),
                "metadata": {"path": "index.json", "sha256": sha256_file(cache / "index.json")},
                "source_dataset_manifest_sha256": dm_sha,
                "array_bytes": {n: (cache / n).stat().st_size for n in ("images.npy", "states.npy", "actions.npy")},
            },
            "sidecars": {
                "path": rel(side),
                "metadata": {"path": "manifest.json", "sha256": sha256_file(side / "manifest.json")},
                "source_memory_manifest_sha256": cm["source_memory_manifest_sha256"],
                "splits": {k: len(v) for k, v in cm["splits"].items()},
            },
            "normalization": {"path": rel(stats_path), "sha256": sha256_file(stats_path), "frozen": True},
        },
        "provenance": {
            "visual_variant": "bpp_v2",
            "description": "PatternLock Easy BPP_v2: v1 identities regenerated live with automatic return-home "
                           "(D1-D6); anchors per data.segment_anchors (sidecar schema v2)",
            "generator_revision": "050f770ec40a37dd39e5ee0e5229ecab98dc2e27+bpp_v2_patch",
            "sim_patch_sha256": "e71d20997956010ead002b6c2699a87f220679374dc8c59f77f7f3aeaf0a47e8",
            "sim_lock_sha256": TREE_LOCK, "sim_tree_manifest_sha256": TREE_MANIFEST,
            "selection_manifest_sha256": json.loads((split_dir / "dataset_manifest.json").read_text())[
                "selection_manifest_sha256"],
            "events_manifest_sha256": cm["source_memory_manifest_sha256"],
            "segment_contract_sha256": SEGMENT_CONTRACT_SHA,
            "stats_choice": a.stats_choice,
            "v1_stats_sha256": "b36c009199956016666d2148f4ff9c1867a7e1c194404f701d9a7b21394fe99d",
            "sha256sums_sha256": sha256_file(split_dir / "SHA256SUMS"),
        },
        "evaluation": {
            "development": {
                "split": "dev", "allocation_seed": 42, "episode_count": 48,
                "selection_manifest_sha256": sha256_file(dev_sel / "manifest.json"),
                "metadata_sha256": sha256_file(dev_sel / "record_dataset_PatternLock_metadata.json"),
                "simulator": "PatternLock bpp_v2=True (auto-return; BenchmarkEnvBuilder extra_env_kwargs)",
            },
            "fresh100": {
                "episode_count": len(f100_recs), "episode_ids": [int(r["episode"]) for r in f100_recs],
                "metadata_sha256": sha256_file(f100),
                "simulator": "PatternLock bpp_v2=True (auto-return), the v1 Fresh100 identities",
            },
        },
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_bytes(out, yaml.safe_dump(manifest, sort_keys=False).encode())
    print(json.dumps({"yaml": str(out), "yaml_sha256": sha256_file(out), "sha256sums_files": nsums,
                      "dataset_manifest_sha256": dm_sha, "stats": rel(stats_path),
                      "stats_sha256": manifest["stages"]["normalization"]["sha256"]}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--src", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--split", required=True, help="name recorded in manifests (train, dev48, pilot_train, ...)")
    b.add_argument("--split-kind", required=True, help="generator split of the statuses (train|dev48)")
    b.add_argument("--selection", required=True)
    b.add_argument("--v1-splits", help="v1 counter manifest whose dev_train/dev_val membership is reused")
    b.add_argument("--subset", action="store_true", help="allow a subset of the selection (pilot / partial)")
    b.add_argument("--episodes", help="comma list (debug)")
    b.add_argument("--workers", type=int, default=32)
    b.add_argument("--force", action="store_true")
    s = sp.add_parser("stats")
    s.add_argument("--split-dir", required=True)
    s.add_argument("--v1-stats", help="optional previous normalization stats for a coverage comparison")
    s.add_argument("--v1-sha", help="optional expected SHA256 of --v1-stats")
    s.add_argument("--v2-name", default="stats_patternlock_easy_bpp_v2_policy_abs")
    c = sp.add_parser("coverage")
    c.add_argument("--split-dir", required=True)
    c.add_argument("--stats", required=True)
    c.add_argument("--out", required=True)
    y = sp.add_parser("yaml")
    y.add_argument("--root", required=True)
    y.add_argument("--split", required=True)
    y.add_argument("--stats-rel", required=True, help="stats.json path relative to the split dir")
    y.add_argument("--copy-stats-from", help="copy this stats dir to --stats-rel's dir first")
    y.add_argument("--expect-stats-sha")
    y.add_argument("--stats-choice", required=True)
    y.add_argument("--readiness", default="provisional", choices=["provisional", "ready"])
    default_inputs = Path(__file__).resolve().parents[1] / "inputs/selection"
    y.add_argument("--dev-selection-dir", default=str(default_inputs / "dev48/selection"))
    y.add_argument("--fresh100-metadata", default=str(default_inputs / "fresh100/selection/record_dataset_PatternLock_metadata.json"))
    y.add_argument("--out", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "stats": cmd_stats, "coverage": cmd_coverage, "yaml": cmd_yaml}[a.cmd](a)


if __name__ == "__main__":
    main()
