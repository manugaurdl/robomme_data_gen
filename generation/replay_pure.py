#!/usr/bin/env python
"""Open-loop replay check (complements the worker's restore-based alignment gate): fresh env (same seed, bpp_v2), reset,
then feed ONLY the recorded action rows 1..T-1 (no state restore at all). Reports max |qpos - parquet state| over all rows
and bit-equality of both cameras vs the parquet PNGs. Expectation: 0 / all equal (the simulator is deterministic).
usage: CUDA_VISIBLE_DEVICES=<g> venv_sim/bin/python replay_pure.py <out_root> <ctl_dir> <out_json> [obs_mode] [max_eps]
(obs_mode != rgb+depth+segmentation -> physics-only replay, images not compared)
"""
import io
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "simulator"
sys.path.insert(0, str(SRC))
import bpp_v2_simulator as LOCK  # noqa: E402

TREE = LOCK.validate_simulator_v2(SRC)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

LOCK.load_recording_api_v2(SRC)
import robomme.robomme_env  # noqa: E402,F401

ENV_KW = dict(obs_mode="rgb+depth+segmentation", control_mode="pd_joint_pos", render_mode="rgb_array",
              reward_mode="dense", difficulty="easy")
out_root, ctl, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OBS = sys.argv[4] if len(sys.argv) > 4 else ENV_KW["obs_mode"]
ENV_KW["obs_mode"] = OBS
IMG = OBS == "rgb+depth+segmentation"
items = json.loads((ctl / "worklist.json").read_text())["items"][: int(sys.argv[5]) if len(sys.argv) > 5 else None]
import time
t2n = lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)  # noqa: E731
res = []
for it in items:
    st = json.loads((out_root / it["split"] / "status" / f"episode_{it['episode']:06d}.json").read_text())
    t = pq.read_table(st["parquet"])
    T = t.num_rows
    state = np.asarray(t.column("state").to_pylist(), dtype=np.float32)
    action = np.asarray(t.column("actions").to_pylist(), dtype=np.float32)
    kinds = t.column("segment_kind").to_pylist()
    dec = lambda col: [np.asarray(Image.open(io.BytesIO(x["bytes"])).convert("RGB")) for x in t.column(col).to_pylist()]  # noqa: E731
    front, wrist = dec("image"), dec("wrist_image")
    rec = it["episode"] if it.get("failure_recovery_episode") is None else int(it["failure_recovery_episode"])
    extra = {"robomme_failure_recovery": True, "robomme_failure_recovery_mode": "z" if rec <= 2 else "xy"} if rec <= 5 else {}
    t0 = time.time()
    env = gym.make("PatternLock", seed=int(it["seed"]), bpp_v2=True, **ENV_KW, **extra)
    obs, info = env.reset()
    base = env.unwrapped
    qd, eq, kbad = [], 0, 0
    if IMG:
        sd = obs["sensor_data"]
        eq += int(np.array_equal(t2n(sd["base_camera"]["rgb"][0]), front[0]) and np.array_equal(t2n(sd["hand_camera"]["rgb"][0]), wrist[0]))
    qd.append(float(np.max(np.abs(t2n(base.agent.robot.qpos[0]) - state[0, :7]))))
    for i in range(1, T):
        obs, r, term, trunc, info = env.step(torch.as_tensor(action[i, :7][None]))
        qd.append(float(np.max(np.abs(t2n(base.agent.robot.qpos[0]).astype(np.float32) - state[i, :7]))))
        if IMG:
            sd = obs["sensor_data"]
            eq += int(np.array_equal(t2n(sd["base_camera"]["rgb"][0]), front[i]) and np.array_equal(t2n(sd["hand_camera"]["rgb"][0]), wrist[i]))
        kbad += int(str(info.get("bpp_v2_segment_kind")) != kinds[i])
    summ = base.bpp_v2_summary()
    env.close()
    r = {"split": it["split"], "episode": it["episode"], "T": T, "qpos_max_abs_diff": max(qd), "rows_qpos_exact": sum(d == 0 for d in qd),
         "images_equal": eq, "segment_kind_mismatch": kbad, "success": bool(summ["success"]),
         "overridden_caller_actions": int(summ["overridden_caller_actions"]), "obs_mode": OBS, "seconds": round(time.time() - t0, 2)}
    print(json.dumps(r), flush=True)
    res.append(r)
agg = {"tree": {k: TREE[k] for k in ("lock_sha256", "tree_manifest_sha256")}, "episodes": len(res),
       "rows": sum(r["T"] for r in res), "qpos_max_abs_diff": max(r["qpos_max_abs_diff"] for r in res),
       "rows_qpos_exact": sum(r["rows_qpos_exact"] for r in res), "images_equal": sum(r["images_equal"] for r in res),
       "segment_kind_mismatch": sum(r["segment_kind_mismatch"] for r in res), "success": sum(r["success"] for r in res),
       "overridden_caller_actions": sum(r["overridden_caller_actions"] for r in res), "per_episode": res}
out_json.write_text(json.dumps(agg, indent=1))
print(json.dumps({k: v for k, v in agg.items() if k != "per_episode"}, indent=1))
