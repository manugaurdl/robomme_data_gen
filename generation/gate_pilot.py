#!/usr/bin/env python
"""P2 pilot gates (independent re-checks on the published pilot outputs + aggregation of the per-episode status).

  success            every work item has a PASS status, 0 failures
  alignment          replay action[t+1] from state[t] -> state[t+1] (worker replay; max |dq| over policy transitions),
                     + negative control (action[t] instead of action[t+1]) to show the gate has power
  home identity      every home row of every episode: front + wrist PNG-decoded arrays bit-equal to ONE reference,
                     state == HOME_QPOS, action == HOME
  contacts           independent recount from the traces: reach rows touch only their target, return rows only the
                     just-pressed button, home rows nothing (incl. exempt previous-button touches)
  red pixels         R>220, G<90, B<90 over every frame of both cameras
  sizes / timing     frames/episode, bytes/episode, s/episode -> projections for 5,000 + 48
usage: venv_sim/bin/python gate_pilot.py <out_root> <ctl_dir> <out_json>
"""
import collections
import gzip
import hashlib
import io
import json
import statistics as S
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# == robomme_env/utils/bpp_v2.py::HOME_QPOS (full precision, float32); also cross-checked vs each trace's summary
HOME_QPOS = np.asarray([-0.00147877566, 0.110702187, 0.00146709458, -2.33247628, -0.000253323766, 2.44317490,
                        0.785576286], dtype=np.float32)
out_root, ctl, out_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
wl = json.loads((ctl / "worklist.json").read_text())
items = wl["items"]
res = {"out_root": str(out_root), "items": len(items)}

# ---- success
st = {}
for it in items:
    p = out_root / it["split"] / "status" / f"episode_{it['episode']:06d}.json"
    if p.exists():
        st[(it["split"], it["episode"])] = json.loads(p.read_text())
fails = sorted(str(p) for p in out_root.glob("*/failed/*.json"))
final_fail = sorted(p.name for p in (ctl / "failed").glob("*.json"))
res["success"] = {"pass_status": len(st), "of": len(items), "failed_attempt_files": len(fails),
                  "final_failures": final_fail, "all_status_pass": all(s["status"] == "PASS" for s in st.values()),
                  "pass": len(st) == len(items) and not final_fail and all(s["status"] == "PASS" for s in st.values())}
trees = {json.dumps(s["tree"], sort_keys=True) for s in st.values()}
res["tree_ids"] = [json.loads(t) for t in trees]
res["worker_sha256"] = sorted({s["worker_sha256"] for s in st.values()})

# ---- alignment (worker replay) + negative control
al = [s["gates"]["alignment"] for s in st.values()]
nc = [s["gates"].get("alignment_negative_control") for s in st.values() if s["gates"].get("alignment_negative_control")]
res["alignment"] = {
    "episodes": len(al), "policy_transitions": sum(a["policy_transitions"] for a in al),
    "policy_max_abs_diff": max(a["policy_max_abs_diff"] for a in al),
    "sim_transitions": sum(a["sim_transitions"] for a in al), "sim_max_abs_diff": max(a["sim_max_abs_diff"] for a in al),
    "row0_max_abs_diff": max(a["row0_qpos_max_abs_diff"] for a in al),
    "sequential_no_restore_qpos_max_abs_diff": max(a["sequential_qpos_max_abs_diff_before_restore"] for a in al),
    "sim_applied_vs_recorded_action_max_abs_diff": max(a["sim_applied_vs_recorded_action_max_abs_diff"] for a in al),
    "segment_kind_divergence": sum(a["diverged_at"] is not None for a in al),
    "replay_success": sum(a["replay_success"] for a in al),
    "images_equal": sum(a["images_equal"] for a in al), "images_total": sum(a["images_total"] for a in al),
    "images_mismatch_max_abs_px": max(a.get("images_mismatch_max_abs_px", 0) for a in al),
    "images_mismatch_pixels_total": sum(a.get("images_mismatch_pixels", 0) for a in al),
    "per_step_policy_motion_median": S.median(a["policy_step_motion_median"] for a in al),
    "tolerance": 1e-5, "all_episode_ok": all(a["ok"] for a in al),
}
if nc:
    res["alignment"]["negative_control_shifted_action"] = {
        "episodes": len(nc), "policy_transitions_before_divergence": sum(a["policy_transitions"] for a in nc),
        "max_abs_diff": max(a["policy_max_abs_diff"] for a in nc),
        "median_over_episodes_of_mean_step_diff": S.median(a["policy_mean_abs_diff_max"] for a in nc),
        "min_over_episodes_of_mean_step_diff": min(a["policy_mean_abs_diff_max"] for a in nc)}
a = res["alignment"]
a["pass"] = bool(a["all_episode_ok"] and a["policy_max_abs_diff"] <= 1e-5 and a["row0_max_abs_diff"] == 0
                 and a["segment_kind_divergence"] == 0 and a["replay_success"] == len(al))

# ---- independent parquet/trace checks
ref_f = ref_w = None
home_rows = home_bad = home_state_bad = 0
home_hashes = collections.Counter()
red_total = 0
frames_total = 0
wrong = collections.Counter()
touch = collections.Counter()
per_ep = []
for (split, e), s in sorted(st.items()):
    t = pq.read_table(s["parquet"])
    T = t.num_rows
    kinds = t.column("segment_kind").to_pylist()
    seg = t.column("segment_id").to_pylist()
    acc = t.column("accepted_event").to_pylist()
    state = np.asarray(t.column("state").to_pylist(), dtype=np.float32)
    action = np.asarray(t.column("actions").to_pylist(), dtype=np.float32)
    fr = t.column("image").to_pylist()
    wr = t.column("wrist_image").to_pylist()
    red = 0
    for i in range(T):
        f = np.asarray(Image.open(io.BytesIO(fr[i]["bytes"])).convert("RGB"))
        w = np.asarray(Image.open(io.BytesIO(wr[i]["bytes"])).convert("RGB"))
        for img in (f, w):
            red += int(np.count_nonzero((img[..., 0] > 220) & (img[..., 1] < 90) & (img[..., 2] < 90)))
        if kinds[i] == "home":
            home_rows += 1
            if ref_f is None:
                ref_f, ref_w = f, w
            same = np.array_equal(f, ref_f) and np.array_equal(w, ref_w)
            home_bad += int(not same)
            home_hashes[(hashlib.sha256(f.tobytes()).hexdigest(), hashlib.sha256(w.tobytes()).hexdigest())] += 1
            home_state_bad += int(not (np.array_equal(state[i, :7], HOME_QPOS) and np.array_equal(action[i, :7], HOME_QPOS)))
    red_total += red
    frames_total += T
    # contacts from the trace (independent recount)
    tr = json.loads(gzip.decompress(Path(s["trace"]).read_bytes()))
    assert np.array_equal(np.asarray(tr["bpp_v2_summary"]["home_qpos"], dtype=np.float32), HOME_QPOS)
    rows = {int(x["bpp_v2_row"]): x for x in tr["collection_audit_trace"] if "bpp_v2_row" in x}
    pw = s["full_password"]
    seg_t = {seg[r]: acc[r] for r in range(T) if acc[r] >= 0 and seg[r] >= 0}
    last = None
    for i in range(1, T):
        tb = rows[i]["touched_button_ids"]
        k = kinds[i]
        if tb:
            touch[k] += 1
        allowed = ({seg_t.get(seg[i])} if k in ("demo_reach", "exec_reach") else {pw[0]} if k == "handoff_reach"
                   else {last} if k == "return" else set())
        if any(b not in allowed for b in tb):
            wrong[k] += 1
        if acc[i] >= 0:
            last = acc[i]
    per_ep.append({"split": split, "episode": e, "source": s["source"], "pw_len": len(pw), "T": T,
                   "exec_start": s["exec_start_idx"], "bytes": s["parquet_bytes"], "times": s["times_s"],
                   "host": s["host"], "lane": s["lane"], "red": red,
                   "align_max": s["gates"]["alignment"]["policy_max_abs_diff"]})
res["home_identity"] = {"home_rows": home_rows, "rows_not_equal_to_reference": home_bad,
                        "distinct_home_frames": len(home_hashes), "state_or_action_not_home": home_state_bad,
                        "reference_front_sha256": next(iter(home_hashes))[0] if home_hashes else None,
                        "reference_wrist_sha256": next(iter(home_hashes))[1] if home_hashes else None,
                        "pass": home_rows > 0 and home_bad == 0 and home_state_bad == 0 and len(home_hashes) == 1}
res["contacts"] = {"wrong_contact_rows_by_kind": dict(wrong), "rows_with_any_contact_by_kind": dict(touch),
                   "pass": sum(wrong.values()) == 0}
res["red_pixels"] = {"frames": frames_total, "cameras": 2, "red_pixels": red_total, "pass": red_total == 0}

# ---- sizes / timing / projections
Ts = [p["T"] for p in per_ep]
by_src = collections.defaultdict(list)
for p in per_ep:
    by_src[p["source"]].append(p)
bytes_per_frame = sum(p["bytes"] for p in per_ep) / sum(Ts)
tot = [p["times"]["total"] for p in per_ep]
rep = [p["times"].get("replay", 0) for p in per_ep]
res["stats"] = {
    "frames_per_episode": {"min": min(Ts), "median": S.median(Ts), "mean": round(S.mean(Ts), 1), "max": max(Ts)},
    "frames_by_source_mean": {k: round(S.mean(p["T"] for p in v), 1) for k, v in by_src.items()},
    "frames_by_pw_len_mean": {str(L): round(S.mean(p["T"] for p in per_ep if p["pw_len"] == L), 1)
                              for L in sorted({p["pw_len"] for p in per_ep})},
    "exec_start_mean": round(S.mean(p["exec_start"] for p in per_ep), 1),
    "parquet_bytes_per_episode_mean": round(S.mean(p["bytes"] for p in per_ep)),
    "parquet_bytes_per_frame": round(bytes_per_frame),
    "seconds_per_episode_per_lane": {"median": S.median(tot), "mean": round(S.mean(tot), 1), "max": max(tot),
                                     "replay_incl_negative_control_mean": round(S.mean(rep), 1),
                                     "sim_mean": round(S.mean(p["times"]["sim"] for p in per_ep), 1),
                                     "export_mean": round(S.mean(p["times"]["export"] for p in per_ep), 1)},
}
res["per_episode"] = per_ep
out_json.write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k != "per_episode"}, indent=1, default=str))
