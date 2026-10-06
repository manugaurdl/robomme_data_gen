#!/usr/bin/env python
"""BPP_v2 generation worker -- ONE long-lived process per lane (amortizes the ~4-5 min NFS import start-up).

Used unchanged for the P2 pilot, the per-node dry runs and the P4 production run.

Loop: claim the next episode of the run's work list (O_EXCL file in the shared control dir on /home, so lanes on both
nodes self-balance), record it with the v2 recorder (exactly S1's path: `DG._run_one_episode` + extra_env_kwargs
{"bpp_v2": True}, the B5K route variant + collection audit trace, RecordWrapper video hooks as B5K), export it
(`export_h5` + `write_patternlock_episode_parquet.main` in-process), run the per-episode gates, replay it for the
(state, action) alignment gate, and publish atomically on node-local disk:
  <out>/<split>/robomme_data_lerobot/data/chunk-XXX/episode_XXXXXX.parquet  (only after every gate passed)
  <out>/<split>/traces/episode_XXXXXX.json.gz    (bpp_v2 summary, audit trace, route/planner logs)
  <out>/<split>/rowlog/episode_XXXXXX.npz        (per-row qpos, qvel, tcp: the replay needs qvel, not in the parquet)
  <out>/<split>/status/episode_XXXXXX.json       (PASS status; written after the parquet)
  <ctl>/done/<split>_<episode>                   (global done marker; resume = skip)
Failures: <out>/<split>/failed/episode_XXXXXX.attempt<k>.json; <ctl>/attempts/<split>_<ep> counts attempts across
processes/nodes; after max_attempts (3) -> <ctl>/failed/<split>_<ep>.json and the lane moves on.
Every status/failure JSON carries the frozen sim tree hash (lock sha + tree manifest sha) validated at start-up.

Exit: 0 = no work left or STOP file; 3 = per-process episode cap reached (lane supervisor relaunches); else crash.
usage: CUDA_VISIBLE_DEVICES=<g> venv_sim/bin/python gen_worker.py --ctl <ctl_dir> --lane <name> [--max-episodes N]
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import resource
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "simulator"
WORKER_FILE = Path(__file__).resolve()


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n").encode())


def _json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def excl_create(path: Path, text: str) -> bool:
    """Atomic create-if-absent (O_EXCL; atomic on NFSv3+). True iff this call created it."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return True


# --------------------------------------------------------------------------------------------------------------- #
# start-up: validate the frozen tree BEFORE importing the simulator, then import from it
# --------------------------------------------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("--ctl", type=Path, required=True)
ap.add_argument("--lane", required=True)
ap.add_argument("--max-episodes", type=int, default=0, help="per-process cap (0 = unlimited)")
ap.add_argument("--only", default="", help="debug: comma list split:episode to run (ignores claims order)")
ARGS = ap.parse_args()
CTL = ARGS.ctl.resolve()
CFG = json.loads((CTL / "config.json").read_text())
WORK = json.loads((CTL / "worklist.json").read_text())["items"]
OUT = Path(CFG["out_root"])
HOST = socket.gethostname().split(".")[0]
T_START = time.time()

sys.path.insert(0, str(SRC))
import bpp_v2_simulator as LOCK  # noqa: E402  (stdlib only)

TREE = LOCK.validate_simulator_v2(SRC)  # raises on any drift from the lock
if CFG.get("expected_tree_manifest_sha256") and TREE["tree_manifest_sha256"] != CFG["expected_tree_manifest_sha256"]:
    raise SystemExit(f"tree manifest {TREE['tree_manifest_sha256']} != expected {CFG['expected_tree_manifest_sha256']}")
if CFG.get("expected_lock_sha256") and TREE["lock_sha256"] != CFG["expected_lock_sha256"]:
    raise SystemExit(f"lock sha {TREE['lock_sha256']} != expected {CFG['expected_lock_sha256']}")
TREE_ID = {"lock_sha256": TREE["lock_sha256"], "tree_manifest_sha256": TREE["tree_manifest_sha256"]}
WORKER_SHA = sha256_file(WORKER_FILE)
log(f"lane {ARGS.lane} host {HOST} pid {os.getpid()} gpu {os.environ.get('CUDA_VISIBLE_DEVICES')} tree {TREE_ID} "
    f"worker {WORKER_SHA[:12]}; importing simulator ...")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
sys.path.insert(0, str(SRC / "memory_diffusion_policy"))
import gymnasium as gym  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

DG, _info = LOCK.load_recording_api_v2(SRC)  # tests._shared.dataset_generation from the validated tree only
from robomme.robomme_env.utils import bpp_v2 as B  # noqa: E402
from robomme.env_record_wrapper import RobommeRecordWrapper  # noqa: E402
import importlib.util  # noqa: E402


def _load_file(name, rel):
    spec = importlib.util.spec_from_file_location(name, SRC / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, spec


B500, _ = _load_file("bpp_v2_b500", "memory_diffusion_policy/scripts/build_patternlock_easy_500_dataset.py")
export_h5, password_from_seed = B500.export_h5, B500.password_from_seed

WRITER, _spec = _load_file("bpp_v2_parquet_writer", "memory_diffusion_policy/scripts/write_patternlock_episode_parquet.py")

from segment_contract import segment_anchor_expectations as SEGMENT_ANCHOR_EXPECTATIONS  # noqa: E402
SEGMENT_CONTRACT_SHA = sha256_file(Path(__file__).with_name("segment_contract.py"))

# Recorder hooks exactly as B5K.record / B500.record (HDF5 buffering lives in the video branch; no video files).
RobommeRecordWrapper._video_should_record = lambda self, task: task != "NO RECORD"
RobommeRecordWrapper._video_append_step_frame = lambda self, *a, **k: None
RobommeRecordWrapper._video_flush_episode_files = lambda self, *a, **k: None

ENV_KW = dict(obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos", render_mode="rgb_array",
              reward_mode="dense", difficulty="easy")
TEMPLATE = Path(CFG["template_parquet"])
TEMPLATE_SHA = sha256_file(TEMPLATE)
if CFG.get("template_sha256") and TEMPLATE_SHA != CFG["template_sha256"]:
    raise SystemExit(f"template parquet sha {TEMPLATE_SHA} != {CFG['template_sha256']}")
HOME_QPOS = np.asarray(B.HOME_QPOS, dtype=np.float32)
log(f"imports done in {time.time() - T_START:.0f}s")


def t2n(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------------------------------------------------- #
# recording
# --------------------------------------------------------------------------------------------------------------- #

def recovery_kwargs(item) -> dict:
    """DG's fixture switches keyed by the source episode (no-op for PatternLock; passed identically for fidelity)."""
    rec = item["episode"] if item.get("failure_recovery_episode") is None else int(item["failure_recovery_episode"])
    if rec <= 5:
        return {"robomme_failure_recovery": True, "robomme_failure_recovery_mode": "z" if rec <= 2 else "xy"}
    return {}


def record(item, work: Path):
    episode, seed = int(item["episode"]), int(item["seed"])
    route = ({"schema": "patternlock_route_variant_v1", "segments": item["route_segments"]}
             if item["source"] == "generated_route_v1" else None)
    rows, holder = [], {}

    def snap(base):
        meta = base.bpp_v2_last_row()
        robot = base.agent.robot
        rows.append({
            "row": int(meta["row"]), "kind": str(meta["segment_kind"]), "owner": str(meta["control_owner"]),
            "seg": int(meta["segment_id"]), "acc": int(meta["accepted_event"]),
            "applied": np.asarray(meta["applied_action"], dtype=np.float32).reshape(-1)[:7].copy(),
            "qpos": t2n(robot.qpos[0]).astype(np.float32).copy(),
            "qvel": t2n(robot.qvel[0]).astype(np.float32).copy(),
            "tcp": t2n(base.agent.tcp.pose.p[0]).astype(np.float64).copy(),
        })

    def configure(base):
        base.patternlock_route_variant = route
        base.patternlock_route_log = []
        base._patternlock_route_segment_index = 0
        base.patternlock_collection_audit_enabled = True
        base.patternlock_collection_audit_trace = []
        holder["base"] = base
        orig_step, orig_reset = base.step, base.reset

        def reset(*a, **k):
            out = orig_reset(*a, **k)
            snap(base)
            return out

        def step(a):
            out = orig_step(a)
            snap(base)
            return out
        base.step, base.reset = step, reset

    meta: dict = {}
    case = DG.DatasetCase("PatternLock", episode, seed, "easy", False, "bpp_v2_gen",
                          failure_recovery_episode=item.get("failure_recovery_episode"))
    ok = DG._run_one_episode(case, seed, work, configure_env=configure, result_metadata=meta,
                             extra_env_kwargs={"bpp_v2": True})
    return bool(ok), rows, meta


# --------------------------------------------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------------------------------------------- #

def decode_png(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def red_count(img: np.ndarray) -> int:
    return int(np.count_nonzero((img[..., 0] > 220) & (img[..., 1] < 90) & (img[..., 2] < 90)))


def arr_sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def gates_on_parquet(item, parquet: Path, rows, meta, exported):
    """Per-episode gates on the written parquet (what the trainer reads) vs the live recording."""
    g, fails = {}, []
    t = pq.read_table(parquet)
    T = t.num_rows
    state = np.asarray(t.column("state").to_pylist(), dtype=np.float32)
    action = np.asarray(t.column("actions").to_pylist(), dtype=np.float32)
    kinds = t.column("segment_kind").to_pylist()
    owner = t.column("control_owner").to_pylist()
    seg = np.asarray(t.column("segment_id").to_pylist(), dtype=np.int64)
    acc = np.asarray(t.column("accepted_event").to_pylist(), dtype=np.int64)
    is_demo = np.asarray(t.column("is_demo").to_pylist(), dtype=bool)
    exec_start = int(t.column("exec_start_idx")[0].as_py())
    pw = [int(x) for x in item["full_password"]]
    L = len(pw) - 1

    # (1) parquet == live recording
    live_ok = (T == len(rows) == int(exported["num_frames"]) and [r["row"] for r in rows] == list(range(T))
               and np.array_equal(state[:, :7], np.stack([r["qpos"] for r in rows])) and np.all(state[:, 7] == 0)
               and np.array_equal(action[:, :7], np.stack([r["applied"] for r in rows])) and np.all(action[:, 7] == -1)
               and kinds == [r["kind"] for r in rows] and owner == [r["owner"] for r in rows]
               and seg.tolist() == [r["seg"] for r in rows] and acc.tolist() == [r["acc"] for r in rows])
    g["parquet_equals_live"] = bool(live_ok)
    if not live_ok:
        fails.append("parquet_equals_live")

    # (2) events: demo pw[0..m] -> handoff pw[0] -> exec pw[1..m]; structure contact -> 20 return -> 1 home
    ev = [(int(r), int(acc[r]), kinds[r]) for r in np.flatnonzero(acc >= 0)]
    want = ([(b, B.KIND_DEMO_REACH) for b in pw] + [(pw[0], B.KIND_HANDOFF_REACH)]
            + [(b, B.KIND_EXEC_REACH) for b in pw[1:]])
    events_ok = [(b, k) for _, b, k in ev] == want
    summ = meta.get("bpp_v2_summary", {})
    events_ok = events_ok and [(e["button"], e["reach_kind"]) for e in summ.get("events", [])] == want \
        and [e["row"] for e in summ.get("events", [])] == [r for r, _, _ in ev]
    g["events"] = [[r, b, k] for r, b, k in ev]
    g["events_ok"] = bool(events_ok)
    if not events_ok:
        fails.append("events")
    struct_ok = kinds[0] == B.KIND_HOME and owner[0] == B.OWNER_SIM and kinds[-1] == B.KIND_HOME
    for r, _, _ in ev:
        seq = kinds[r + 1:r + 22]
        struct_ok &= seq == [B.KIND_RETURN] * B.RETURN_STEPS + [B.KIND_HOME]
        struct_ok &= all(owner[i] == B.OWNER_SIM and seg[i] == -1 for i in range(r + 1, r + 22))
    struct_ok &= (ev[-1][0] + 21 == T - 1) if ev else False
    struct_ok &= bool(np.array_equal(is_demo, np.arange(T) < exec_start))
    handoff_ok = len(ev) > len(pw) and ev[len(pw)][2] == B.KIND_HANDOFF_REACH and ev[len(pw)][0] + 21 == exec_start
    struct_ok &= bool(kinds[exec_start] == B.KIND_HOME and handoff_ok)
    home_rows = [i for i, k in enumerate(kinds) if k == B.KIND_HOME]
    struct_ok &= bool(np.all(action[home_rows, :7] == HOME_QPOS[None]))
    struct_ok &= bool(summ.get("success")) and not summ.get("failed") and summ.get("rows") == T
    g["structure_ok"] = bool(struct_ok)
    if not struct_ok:
        fails.append("structure")
    g["overridden_caller_actions"] = int(summ.get("overridden_caller_actions", -1))
    if g["overridden_caller_actions"] != 0:
        fails.append("overridden_caller_actions")

    # (3) data contract: logical+1 counter from exec events and within-reach anchors
    own = np.asarray([1 if o == B.OWNER_POLICY else 0 for o in owner], dtype=np.int64)
    exec_rows = [r for r, _, k in ev if k == B.KIND_EXEC_REACH]
    transition = np.zeros(T, dtype=bool)
    for c in exec_rows:
        transition[c + 1] = True
    counter = np.where(np.arange(T) >= exec_start, np.cumsum(transition), -1)
    contract = {}
    try:
        valid, trows, pad = SEGMENT_ANCHOR_EXPECTATIONS({"control_owner": own, "segment_id": seg}, episode=int(item["episode"]),
                                                        num_frames=T, exec_start=exec_start, transition=transition)
        anchors = np.flatnonzero(valid)
        policy_exec = int(own[exec_start:].sum())
        settled = [i for i in home_rows if exec_start <= i < T - 1]
        tgt = trows[anchors]
        tgt_ok = bool(np.all(own[tgt] == 1) and np.all(seg[tgt] == seg[anchors + 1][:, None]) and np.all(tgt > anchors[:, None]))
        # un-padded targets are exactly t+1+j; padded repeat the segment's last row
        raw = anchors[:, None] + 1 + np.arange(16)[None]
        rows_ok = bool(np.all(np.where(pad[anchors], True, tgt == raw)))
        c_ok = (int(counter[-1]) == L and len(exec_rows) == L and counter[exec_start] == 0
                and bool(np.all(counter[anchors] == counter[anchors + 1])) and not np.any(transition & (own == 1)))
        contract = {"ok": bool(len(anchors) == policy_exec and set(settled) <= set(anchors.tolist()) and tgt_ok and rows_ok
                               and c_ok), "anchors": int(len(anchors)), "policy_owned_exec_rows": policy_exec,
                    "settled_home_anchors": settled, "padded_anchors": int(pad[anchors].any(1).sum()),
                    "targets_policy_same_segment": tgt_ok, "unpadded_targets_are_t_plus_1_plus_j": rows_ok,
                    "counter_rise_rows": np.flatnonzero(transition).tolist(), "counter_ok": bool(c_ok)}
    except Exception as exc:
        contract = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    g["contract"] = contract
    if not contract.get("ok"):
        fails.append("contract")

    # (4) contact audit on the collection trace: reach rows may touch only their target; return rows only the
    #     just-pressed button; home rows nothing. (Previous-button touches in exec reaches are exempt in the env but
    #     are counted here as wrong contacts too.)
    trace = meta.get("patternlock_collection_audit_trace", [])
    by_row = {}
    for e in trace:
        if "bpp_v2_row" in e:
            by_row.setdefault(int(e["bpp_v2_row"]), []).append(e)
    trace_ok = sorted(by_row) == list(range(1, T)) and all(len(v) == 1 for v in by_row.values())
    seg_target = {}
    for r, b, k in ev:
        if seg[r] >= 0:
            seg_target[int(seg[r])] = b
    # handoff reach is sim-owned (segment -1): its target is pw[0]; map by kind
    wrong = collections.Counter()
    wrong_rows = []
    touch_rows = collections.Counter()
    last_pressed = None
    ev_by_row = {r: b for r, b, _ in ev}
    for i in range(1, T):
        touched = by_row.get(i, [{}])[0].get("touched_button_ids", [])
        k = kinds[i]
        if touched:
            touch_rows[k] += 1
        if k in (B.KIND_DEMO_REACH, B.KIND_EXEC_REACH):
            allowed = {seg_target.get(int(seg[i]))}
        elif k == B.KIND_HANDOFF_REACH:
            allowed = {pw[0]}
        elif k == B.KIND_RETURN:
            allowed = {last_pressed}
        else:
            allowed = set()
        bad = [b for b in touched if b not in allowed]
        if bad:
            wrong[k] += 1
            wrong_rows.append([i, k, bad])
        if i in ev_by_row:
            last_pressed = ev_by_row[i]
    contacts = {"trace_rows_ok": bool(trace_ok), "wrong_contact_rows": dict(wrong), "wrong_rows": wrong_rows[:20],
                "rows_with_any_contact": dict(touch_rows),
                "ok": bool(trace_ok and sum(wrong.values()) == 0)}
    g["contacts"] = contacts
    if not contacts["ok"]:
        fails.append("contacts")

    # (5) images: red pixels, home identity, and decoded frames kept for the replay image parity
    fr = t.column("image").to_pylist()
    wr = t.column("wrist_image").to_pylist()
    front = [decode_png(x["bytes"]) for x in fr]
    wrist = [decode_png(x["bytes"]) for x in wr]
    red = sum(red_count(a) for a in front) + sum(red_count(a) for a in wrist)
    g["red_pixels"] = int(red)
    if red:
        fails.append("red_pixels")
    hf = {arr_sha(front[i]) for i in home_rows}
    hw = {arr_sha(wrist[i]) for i in home_rows}
    home = {"n_home": len(home_rows), "front_sha256": sorted(hf), "wrist_sha256": sorted(hw),
            "identical_within_episode": len(hf) == 1 and len(hw) == 1,
            "state_eq_home": bool(np.all(state[home_rows, :7] == HOME_QPOS[None])),
            "qvel_zero": bool(all(np.all(rows[i]["qvel"] == 0) for i in home_rows))}
    ref = CFG.get("home_ref")
    if ref:
        home["matches_ref"] = sorted(hf) == [ref["front_sha256"]] and sorted(hw) == [ref["wrist_sha256"]]
    home["ok"] = bool(home["identical_within_episode"] and home["state_eq_home"] and home["qvel_zero"]
                      and home.get("matches_ref", True))
    g["home"] = home
    if not home["ok"]:
        fails.append("home")
    lengths = collections.Counter(kinds)
    info = {"T": T, "exec_start": exec_start, "rows_by_kind": dict(lengths),
            "rows_by_owner": dict(collections.Counter(owner))}
    return g, fails, info, (state, action, owner, kinds, front, wrist)


def replay_alignment(item, rows, parsed, shift: int = 0):
    """(state, action) alignment gate: fresh env (same seed, v2), reset, then for t = 0..T-2 feed action[t+1]; before
    every policy/expert-owned step restore the robot to the RECORDED state[t] (parquet qpos + recorded qvel), step, and
    compare the resulting qpos with parquet state[t+1]. Sim-owned steps are fed the recorded (applied) action too (the env
    applies its schedule; any mismatch counts as an overridden caller action). shift=1 is the negative control
    (feeds action[t] instead of action[t+1] on policy steps)."""
    state, action, owner, kinds, front, wrist = parsed
    T = len(state)
    kw = dict(ENV_KW)
    # physics-only replay (obs_mode "state") is bit-identical in qpos to the rgb one (verified on the pilot) and needs no
    # rendering; images are compared only when the replay renders the recorder's obs mode.
    kw["obs_mode"] = CFG.get("replay_obs_mode", ENV_KW["obs_mode"])
    images = kw["obs_mode"] == ENV_KW["obs_mode"]
    env = gym.make("PatternLock", seed=int(item["seed"]), bpp_v2=True, **kw, **recovery_kwargs(item))
    out = {"shift": shift, "replay_obs_mode": kw["obs_mode"]}
    try:
        obs, info = env.reset()
        base = env.unwrapped
        robot = base.agent.robot
        q0 = t2n(robot.qpos[0]).astype(np.float32)
        out["row0_qpos_max_abs_diff"] = float(np.max(np.abs(q0 - state[0, :7])))
        img_eq = img_tot = 0
        if shift == 0 and images:
            sd = obs["sensor_data"]
            img_tot += 1
            img_eq += int(np.array_equal(t2n(sd["base_camera"]["rgb"][0]), front[0]) and
                          np.array_equal(t2n(sd["hand_camera"]["rgb"][0]), wrist[0]))
        pol_d, sim_d, pre_q, pre_v, kinds_bad, sim_act = [], [], [], [], [], []
        img_max, img_px = 0, 0
        motion = []
        diverged_at = None
        for t in range(T - 1):
            pol = owner[t + 1] == B.OWNER_POLICY
            a = action[t + 1, :7] if not (pol and shift) else action[t + 1 - shift, :7]
            if pol:
                qn = t2n(robot.qpos[0]).astype(np.float32)
                vn = t2n(robot.qvel[0]).astype(np.float32)
                pre_q.append(float(np.max(np.abs(qn - state[t, :7]))))
                pre_v.append(float(np.max(np.abs(vn - rows[t]["qvel"]))))
                dev = robot.qpos.device
                robot.set_qpos(torch.as_tensor(state[t, :7][None], dtype=torch.float32, device=dev))
                robot.set_qvel(torch.as_tensor(rows[t]["qvel"][None], dtype=torch.float32, device=dev))
            obs, rew, term, trunc, info = env.step(torch.as_tensor(a[None], dtype=torch.float32))
            q = t2n(robot.qpos[0]).astype(np.float32)
            d = float(np.max(np.abs(q - state[t + 1, :7])))
            (pol_d if pol else sim_d).append(d)
            if pol:
                motion.append(float(np.max(np.abs(state[t + 1, :7] - state[t, :7]))))
            if not pol:
                # the sim applied its own scheduled target: compare it with the recorded (applied) action row
                ap_ = np.asarray(info.get("bpp_v2_applied_action"), dtype=np.float32).reshape(-1)[:7]
                sim_act.append(float(np.max(np.abs(ap_ - action[t + 1, :7]))))
            k = str(info.get("bpp_v2_segment_kind"))
            if k != kinds[t + 1]:
                kinds_bad.append(t + 1)
                if diverged_at is None:
                    diverged_at = t + 1
                if shift:
                    break
            if shift == 0 and images:
                sd = obs["sensor_data"]
                f_, w_ = t2n(sd["base_camera"]["rgb"][0]), t2n(sd["hand_camera"]["rgb"][0])
                img_tot += 1
                same = np.array_equal(f_, front[t + 1]) and np.array_equal(w_, wrist[t + 1])
                img_eq += int(same)
                if not same:
                    df = np.abs(f_.astype(np.int16) - front[t + 1]); dw = np.abs(w_.astype(np.int16) - wrist[t + 1])
                    img_max = max(img_max, int(df.max()), int(dw.max()))
                    img_px += int(np.count_nonzero(df)) + int(np.count_nonzero(dw))
        summ = base.bpp_v2_summary()
        out.update({
            "policy_transitions": len(pol_d), "policy_max_abs_diff": max(pol_d, default=0.0),
            "policy_mean_abs_diff_max": float(np.mean(pol_d)) if pol_d else 0.0,
            "sim_transitions": len(sim_d), "sim_max_abs_diff": max(sim_d, default=0.0),
            "sequential_qpos_max_abs_diff_before_restore": max(pre_q, default=0.0),
            "sequential_qvel_max_abs_diff_before_restore": max(pre_v, default=0.0),
            "policy_step_motion_median": float(np.median(motion)) if motion else 0.0,
            "policy_step_motion_max": max(motion, default=0.0),
            "segment_kind_mismatch_rows": kinds_bad[:10], "diverged_at": diverged_at,
            "sim_applied_vs_recorded_action_max_abs_diff": max(sim_act, default=0.0),
            "overridden_caller_actions_ulp_level": int(summ["overridden_caller_actions"]),
            "replay_success": bool(summ["success"]), "images_equal": img_eq, "images_total": img_tot,
            "images_mismatch_max_abs_px": img_max, "images_mismatch_pixels": img_px,
        })
    finally:
        try:
            env.close()
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------------------------------------------- #
# episode driver
# --------------------------------------------------------------------------------------------------------------- #

def paths(item):
    split, ep = item["split"], int(item["episode"])
    root = OUT / split
    return {
        "parquet": root / "robomme_data_lerobot/data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet",
        "status": root / "status" / f"episode_{ep:06d}.json",
        "trace": root / "traces" / f"episode_{ep:06d}.json.gz",
        "rowlog": root / "rowlog" / f"episode_{ep:06d}.npz",
        "failed": root / "failed",
    }


def key(item) -> str:
    return f"{item['split']}_{int(item['episode']):06d}"


def run_episode(item, attempt: int) -> dict:
    ep, seed = int(item["episode"]), int(item["seed"])
    if tuple(password_from_seed(seed)) != tuple(item["full_password"]):
        raise ValueError(f"episode {ep}: seed/password drift")
    P = paths(item)
    work = OUT / "work" / ARGS.lane / key(item)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    times = {}
    t0 = time.time()
    ok, rows, meta = record(item, work)
    times["sim"] = time.time() - t0
    if not ok:
        raise RuntimeError(f"scripted recorder did not terminate successfully (bpp_v2_summary="
                           f"{json.dumps(meta.get('bpp_v2_summary', {}), default=_json_default)[:400]})")
    t1 = time.time()
    h5 = work / "hdf5_files" / f"PatternLock_ep{ep}_seed{seed}.h5"
    npz = work / f"episode_{ep:06d}.npz"
    exported = export_h5(h5, ep, npz)
    wpq = work / f"episode_{ep:06d}.parquet"
    conv = work / "converter_status.json"
    argv = sys.argv
    sys.argv = [str(_spec.origin), "--input", str(npz), "--template", str(TEMPLATE), "--output", str(wpq),
                "--episode", str(ep), "--status-out", str(conv)]
    try:
        rc = WRITER.main()
    finally:
        sys.argv = argv
    if rc != 0:
        raise RuntimeError(f"parquet writer rc={rc}")
    converted = json.loads(conv.read_text())
    times["export"] = time.time() - t1
    t2 = time.time()
    gates, fails, info, parsed = gates_on_parquet(item, wpq, rows, meta, exported)
    times["checks"] = time.time() - t2
    if CFG.get("replay", True):
        t3 = time.time()
        al = replay_alignment(item, rows, parsed)
        tol = CFG.get("align_tol", 1e-5)
        al["ok"] = bool(al["policy_max_abs_diff"] <= tol and al["row0_qpos_max_abs_diff"] == 0
                        and al["sim_max_abs_diff"] <= tol and al["sim_applied_vs_recorded_action_max_abs_diff"] <= tol
                        and al["diverged_at"] is None and al["replay_success"])
        gates["alignment"] = al
        if not al["ok"]:
            fails.append("alignment")
        if CFG.get("negative_control"):
            gates["alignment_negative_control"] = replay_alignment(item, rows, parsed, shift=1)
        times["replay"] = time.time() - t3
    del parsed
    if fails:
        raise RuntimeError(f"episode gates failed: {fails}; " + json.dumps(
            {k: gates.get(k) for k in ("events_ok", "structure_ok", "contract", "contacts", "red_pixels", "home",
                                       "alignment", "overridden_caller_actions")}, default=_json_default)[:1500])
    # publish: trace + rowlog, then the parquet (atomic rename on the same fs), then the status
    trace_payload = {"schema": "bpp_v2_generation_trace_v1", "split": item["split"], "episode": ep, "seed": seed,
                     "bpp_v2_summary": meta.get("bpp_v2_summary"), "route_log": meta.get("patternlock_route_log"),
                     "planner_attempt_log": meta.get("planner_attempt_log"),
                     "collection_audit_trace": meta.get("patternlock_collection_audit_trace")}
    atomic_write_bytes(P["trace"], gzip.compress(json.dumps(trace_payload, default=_json_default).encode(), 6))
    bio = io.BytesIO()
    np.savez(bio, qpos=np.stack([r["qpos"] for r in rows]), qvel=np.stack([r["qvel"] for r in rows]),
             tcp=np.stack([r["tcp"] for r in rows]))
    atomic_write_bytes(P["rowlog"], bio.getvalue())
    P["parquet"].parent.mkdir(parents=True, exist_ok=True)
    os.replace(wpq, P["parquet"])
    pq_sha = sha256_file(P["parquet"])
    if pq_sha != converted["parquet_sha256"]:
        raise RuntimeError("published parquet hash differs from the writer's")
    times["total"] = time.time() - t0
    plog = meta.get("planner_attempt_log") or []
    status = {
        "status": "PASS", "schema": "bpp_v2_generation_status_v1", "split": item["split"], "episode": ep, "seed": seed,
        "source": item["source"], "full_password": item["full_password"], "variant_index": item.get("variant_index"),
        "route_segments": item.get("route_segments"), "base_episode_index": item.get("base_episode_index"),
        "failure_recovery_episode": item.get("failure_recovery_episode"),
        "attempt": attempt, "host": HOST, "lane": ARGS.lane, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(), "tree": TREE_ID, "worker_sha256": WORKER_SHA, "segment_contract_sha256": SEGMENT_CONTRACT_SHA,
        "template_parquet": str(TEMPLATE), "template_sha256": TEMPLATE_SHA, "run": CFG.get("run"),
        "num_frames": info["T"], "exec_start_idx": info["exec_start"], "rows_by_kind": info["rows_by_kind"],
        "rows_by_owner": info["rows_by_owner"], "execution_password_ids": item["full_password"][1:],
        "parquet": str(P["parquet"]), "parquet_sha256": pq_sha, "parquet_bytes": P["parquet"].stat().st_size,
        "trace": str(P["trace"]), "trace_sha256": sha256_file(P["trace"]),
        "rowlog": str(P["rowlog"]), "rowlog_sha256": sha256_file(P["rowlog"]),
        "planner_aborts": (meta.get("bpp_v2_summary") or {}).get("planner_aborts"),
        "planner_calls": len(plog), "planner_fallback_calls": sum(bool(x.get("used_fallback")) for x in plog),
        "route_log_len": len(meta.get("patternlock_route_log") or []),
        "gates": gates, "times_s": {k: round(v, 3) for k, v in times.items()}, "finished_at": now(),
    }
    atomic_json(P["status"], status)
    shutil.rmtree(work, ignore_errors=True)
    return status


def claims_snapshot():
    try:
        claimed = set(os.listdir(CTL / "claims"))
    except FileNotFoundError:
        claimed = set()
    return claimed


def next_item(current_file: Path):
    # resume the episode this lane was processing when its previous process died
    if current_file.exists():
        k = current_file.read_text().strip()
        for it in WORK:
            if key(it) == k and not (CTL / "done" / k).exists() and not (CTL / "failed" / f"{k}.json").exists():
                return it, "resumed"
        current_file.unlink(missing_ok=True)
    claimed = claims_snapshot()
    for it in WORK:
        k = key(it)
        if k in claimed:
            continue
        if excl_create(CTL / "claims" / k, json.dumps({"lane": ARGS.lane, "host": HOST, "pid": os.getpid(),
                                                         "at": now()})):
            return it, "claimed"
    return None, None


def heartbeat(state: dict):
    atomic_json(CTL / "lanes" / f"{ARGS.lane}.json", state)


def main():
    for d in ("claims", "done", "failed", "attempts", "lanes"):
        (CTL / d).mkdir(parents=True, exist_ok=True)
    local = OUT / "lanes"
    local.mkdir(parents=True, exist_ok=True)
    current_file = local / f"{ARGS.lane}.current"
    only = [s for s in ARGS.only.split(",") if s]
    hb = {"lane": ARGS.lane, "host": HOST, "pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "started": now(), "import_s": round(time.time() - T_START, 1), "done": 0, "failed_attempts": 0,
          "current": None, "tree": TREE_ID, "worker_sha256": WORKER_SHA}
    heartbeat(hb)
    n = 0
    while True:
        stop = [f for f in ("STOP", f"STOP.{HOST}", f"STOP.{ARGS.lane}") if (CTL / f).exists()]
        if stop:
            log(f"{stop[0]} file present -> exit 0")
            hb.update(current=None, state="stopped", updated=now())
            heartbeat(hb)
            return 0
        if ARGS.max_episodes and n >= ARGS.max_episodes:
            log(f"per-process cap {ARGS.max_episodes} reached -> exit 3")
            hb.update(current=None, state="cap", updated=now())
            heartbeat(hb)
            return 3
        if ARGS.only:
            if not only:
                log("--only list done -> exit 0")
                return 0
            it = next((w for w in WORK if f"{w['split']}:{w['episode']}" == only[0]), None)
            only.pop(0)
            how = "only"
            if it is None:
                continue
        else:
            it, how = next_item(current_file)
        if it is None:
            log("no unclaimed work left -> exit 0")
            hb.update(current=None, state="no_work", updated=now())
            heartbeat(hb)
            return 0
        k = key(it)
        current_file.write_text(k)
        att_file = CTL / "attempts" / k
        attempt = int(att_file.read_text() or 0) + 1 if att_file.exists() else 1
        maxa = int(CFG.get("max_attempts", 3))
        if attempt > maxa:
            atomic_json(CTL / "failed" / f"{k}.json", {"key": k, "attempts": attempt - 1, "lane": ARGS.lane,
                                                       "host": HOST, "at": now(), "tree": TREE_ID,
                                                       "note": "attempt budget exhausted (previous process died?)"})
            current_file.unlink(missing_ok=True)
            continue
        atomic_write_bytes(att_file, str(attempt).encode())
        hb.update(current=k, attempt=attempt, state="running", updated=now(), rss_mb=round(rss_mb()))
        heartbeat(hb)
        log(f"{k} ({how}) attempt {attempt} seed {it['seed']} pw {it['full_password']} src {it['source']}")
        t0 = time.time()
        try:
            st = run_episode(it, attempt)
            excl_create(CTL / "done" / k, json.dumps({"host": HOST, "lane": ARGS.lane, "parquet_sha256": st["parquet_sha256"],
                                                      "num_frames": st["num_frames"], "at": now()}))
            n += 1
            hb["done"] += 1
            al = st["gates"].get("alignment", {})
            log(f"{k} PASS T={st['num_frames']} exec_start={st['exec_start_idx']} {st['times_s']} "
                f"align_max={al.get('policy_max_abs_diff')} img_eq={al.get('images_equal')}/{al.get('images_total')} "
                f"rss={rss_mb():.0f}MB")
        except Exception as exc:
            hb["failed_attempts"] += 1
            fail = {"status": "FAIL", "key": k, "split": it["split"], "episode": it["episode"], "seed": it["seed"],
                    "attempt": attempt, "host": HOST, "lane": ARGS.lane, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "tree": TREE_ID, "worker_sha256": WORKER_SHA, "error": f"{type(exc).__name__}: {exc}"[:4000],
                    "traceback": traceback.format_exc()[-6000:], "seconds": round(time.time() - t0, 1), "at": now()}
            P = paths(it)
            atomic_json(P["failed"] / f"episode_{int(it['episode']):06d}.attempt{attempt}.json", fail)
            log(f"{k} FAIL attempt {attempt}: {fail['error'][:300]}")
            if attempt >= maxa:
                atomic_json(CTL / "failed" / f"{k}.json", fail)
            else:
                # retry right away in this process (bounded by the shared attempt counter)
                hb.update(state="retry", updated=now())
                heartbeat(hb)
                continue
        current_file.unlink(missing_ok=True)
        hb.update(current=None, state="idle", updated=now(), rss_mb=round(rss_mb()),
                  last_episode_s=round(time.time() - t0, 1))
        heartbeat(hb)


if __name__ == "__main__":
    sys.exit(main())
