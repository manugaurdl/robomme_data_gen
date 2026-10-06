#!/usr/bin/env python3
"""BPP_v2 generation control-plane helpers (stdlib only; any python3 works). Used by gen_coordinator.sh.

  gen_ctl.py plan <run> --out-root DIR --durable-root DIR [--splits train,dev48] [--episodes train:0,dev48:1,...]
                  [--pilot] [--no-replay] [--negative-control] [--home-ref FRONT_SHA:WRIST_SHA]
  gen_ctl.py status <run> [--json]          aggregate progress from the shared control dir -> <ctl>/progress.json
  gen_ctl.py retry-failed <run>             clear final failures (+ their attempts/claims) so lanes pick them up again
  gen_ctl.py release-stale <run> [--minutes 20]   drop claims of lanes whose heartbeat is older than N min (dead lanes)

Control dir: $BPP_GEN_CONTROL_ROOT/<run>/ = config.json, worklist.json, claims/, done/,
failed/, attempts/, lanes/, nodes/, STOP, progress.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CTL_ROOT = Path(os.environ.get("BPP_GEN_CONTROL_ROOT", Path.home() / "robomme_data_gen_runs/control"))
INPUTS = Path(os.environ.get("BPP_SELECTION_ROOT", PACKAGE_ROOT / "inputs/selection"))
SELECTION = {"train": INPUTS / "train/selection/manifest.json", "dev48": INPUTS / "dev48/selection/manifest.json"}
TEMPLATE_SRC = Path(os.environ.get("BPP_TEMPLATE_PARQUET", PACKAGE_ROOT / "inputs/template.parquet"))
TEMPLATE_LOCAL = TEMPLATE_SRC
PILOT_SEED = 20260925


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def items_for(split: str) -> list[dict]:
    recs = json.loads(SELECTION[split].read_text())["records"]
    out = []
    for r in recs:
        route = r["source"] == "generated_route_v1"
        out.append({
            "split": split, "episode": int(r["episode"]), "seed": int(r["seed"]), "source": r["source"],
            "full_password": [int(x) for x in r["full_password"]],
            "route_segments": r.get("route_segments") if route else None,
            "variant_index": r.get("variant_index"), "base_episode_index": r.get("base_episode_index"),
            # DG fixture switch owner (B5K: base episode for route variants; B500/preserved: the episode itself)
            "failure_recovery_episode": int(r["base_episode_index"]) if route else None,
        })
    return out


def cmd_plan(a):
    ctl = CTL_ROOT / a.run
    splits = [s for s in a.splits.split(",") if s]
    all_items = {s: items_for(s) for s in splits}
    if a.pilot:
        train = all_items["train"]
        fixed = sorted(it["episode"] for it in train)[:min(4, len(train))]
        rest = sorted(it["episode"] for it in train if it["episode"] not in fixed)
        rnd = sorted(random.Random(PILOT_SEED).sample(rest, min(16, len(rest))))
        dev = sorted(it["episode"] for it in all_items["dev48"])
        want = {"train": fixed + rnd, "dev48": dev[:min(4, len(dev))]}
        items = [it for s in ("dev48", "train") for it in all_items[s] if it["episode"] in want[s]]
        items.sort(key=lambda it: (it["split"] != "dev48", want[it["split"]].index(it["episode"])))
        pilot_info = {"pilot_seed": PILOT_SEED, "train_fixed": fixed, "train_random16": rnd, "dev48": want["dev48"]}
    elif a.episodes:
        pick = [tuple(x.split(":")) for x in a.episodes.split(",") if x]
        idx = {(it["split"], it["episode"]): it for s in splits for it in all_items[s]}
        items = [idx[(s, int(e))] for s, e in pick]
        pilot_info = None
    else:
        # production order: dev48 first (small, lets P5 start its dev48 pass early), then train by episode id
        items = [it for s in ("dev48", "train") if s in all_items for it in sorted(all_items[s], key=lambda x: x["episode"])]
        pilot_info = None
    keys = [f"{it['split']}_{it['episode']:06d}" for it in items]
    assert len(set(keys)) == len(keys), "duplicate work items"
    worklist = {"schema": "bpp_v2_worklist_v1", "run": a.run, "n": len(items), "items": items,
                "selection_sha256": {s: sha256_file(SELECTION[s]) for s in splits}, "pilot": pilot_info}
    cfg = {
        "schema": "bpp_v2_gen_config_v1", "run": a.run, "out_root": a.out_root, "durable_root": a.durable_root,
        "template_parquet": str(TEMPLATE_LOCAL), "template_source": str(TEMPLATE_SRC),
        "template_sha256": sha256_file(TEMPLATE_SRC), "replay": not a.no_replay, "align_tol": 1e-5,
        "negative_control": bool(a.negative_control), "max_attempts": 3, "replay_obs_mode": a.replay_obs_mode,
        "expected_tree_manifest_sha256": a.tree_manifest, "expected_lock_sha256": a.lock_sha,
        "home_ref": ({"front_sha256": a.home_ref.split(":")[0], "wrist_sha256": a.home_ref.split(":")[1]}
                     if a.home_ref else None),
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if (ctl / "worklist.json").exists():
        old = json.loads((ctl / "worklist.json").read_text())
        if old["items"] != items:
            sys.exit(f"{ctl} already has a DIFFERENT worklist; use a new run name")
        print(f"worklist unchanged ({len(items)} items); config rewritten")
    for d in ("claims", "done", "failed", "attempts", "lanes", "nodes"):
        (ctl / d).mkdir(parents=True, exist_ok=True)
    atomic_json(ctl / "worklist.json", worklist)
    atomic_json(ctl / "config.json", cfg)
    print(json.dumps({"ctl": str(ctl), "items": len(items), "splits": {s: sum(1 for it in items if it["split"] == s)
                                                                          for s in splits}, "pilot": pilot_info,
                      "config": cfg}, indent=1))


def _load(ctl):
    wl = json.loads((ctl / "worklist.json").read_text())
    return wl, json.loads((ctl / "config.json").read_text())


def cmd_status(a):
    ctl = CTL_ROOT / a.run
    wl, cfg = _load(ctl)
    items = wl["items"]
    total = {}
    for it in items:
        total[it["split"]] = total.get(it["split"], 0) + 1
    done = {p.name: p.stat().st_mtime for p in (ctl / "done").iterdir()}
    failed = sorted(p.stem for p in (ctl / "failed").glob("*.json"))
    claims = set(os.listdir(ctl / "claims"))
    nowt = time.time()
    by_split = {s: {"total": n, "done": sum(1 for k in done if k.startswith(s + "_")),
                    "failed": sum(1 for k in failed if k.startswith(s + "_"))} for s, n in total.items()}
    in_flight = sorted(claims - set(done) - set(failed))
    lanes = []
    for p in sorted((ctl / "lanes").glob("*.json")):
        try:
            h = json.loads(p.read_text())
        except Exception:
            continue
        h["age_s"] = round(nowt - p.stat().st_mtime)
        lanes.append({k: h.get(k) for k in ("lane", "host", "gpu", "pid", "state", "current", "done", "failed_attempts",
                                            "age_s", "rss_mb", "last_episode_s", "import_s")})
    alive = [l for l in lanes if l["state"] in ("running", "idle", "retry") and l["age_s"] < 900]
    rates = {}
    for win in (5, 15, 60):
        rates[f"per_min_last_{win}m"] = round(sum(1 for t in done.values() if nowt - t < win * 60) / win, 2)
    n_total, n_done, n_failed = len(items), len(done), len(failed)
    remaining = n_total - n_done - n_failed
    r = rates["per_min_last_15m"] or rates["per_min_last_5m"]
    eta = None
    if remaining == 0:
        eta = "complete"
    elif r:
        eta = (dt.datetime.now() + dt.timedelta(minutes=remaining / r)).strftime("%Y-%m-%d %H:%M")
    hosts = {}
    for l in alive:
        hosts[l["host"]] = hosts.get(l["host"], 0) + 1
    prog = {"run": a.run, "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "total": n_total,
            "done": n_done, "failed_final": n_failed, "remaining": remaining, "in_flight": len(in_flight),
            "by_split": by_split, "rates": rates, "eta": eta, "lanes_alive": len(alive), "lanes_alive_by_host": hosts,
            "lanes_total": len(lanes), "stop_file": (ctl / "STOP").exists(), "failed_keys": failed[:50],
            "first_done": min(done.values()) if done else None}
    if prog["first_done"]:
        prog["first_done"] = dt.datetime.fromtimestamp(prog["first_done"]).strftime("%H:%M:%S")
    atomic_json(ctl / "progress.json", prog)
    if a.json:
        print(json.dumps({**prog, "lanes": lanes}, indent=1))
    else:
        print(f"[{a.run}] {prog['at']} done {n_done}/{n_total} failed {n_failed} in-flight {len(in_flight)} "
              f"rate/min 5m={rates['per_min_last_5m']} 15m={rates['per_min_last_15m']} ETA {eta} "
              f"lanes alive {len(alive)}/{len(lanes)} {hosts} STOP={prog['stop_file']}")
        for s, v in by_split.items():
            print(f"  {s}: {v}")
        if failed:
            print("  failed:", failed[:20])
        stale = [l for l in lanes if l["state"] in ("running", "retry") and l["age_s"] >= 900]
        if stale:
            print("  STALE lanes (heartbeat >15 min while running):", [(l["lane"], l["current"], l["age_s"]) for l in stale])


def cmd_retry_failed(a):
    ctl = CTL_ROOT / a.run
    n = 0
    for p in sorted((ctl / "failed").glob("*.json")):
        k = p.stem
        for q in (ctl / "attempts" / k, ctl / "claims" / k, p):
            try:
                q.unlink()
            except FileNotFoundError:
                pass
        n += 1
        print("cleared", k)
    print(f"{n} failures cleared; relaunch lanes (gen_coordinator.sh launch ...) to retry them")


def cmd_release_stale(a):
    ctl = CTL_ROOT / a.run
    done = set(os.listdir(ctl / "done"))
    failed = {p.stem for p in (ctl / "failed").glob("*.json")}
    nowt = time.time()
    live_lanes = set()
    for p in (ctl / "lanes").glob("*.json"):
        if nowt - p.stat().st_mtime < a.minutes * 60:
            live_lanes.add(p.stem)
    n = 0
    for k in sorted(set(os.listdir(ctl / "claims")) - done - failed):
        c = ctl / "claims" / k
        try:
            owner = json.loads(c.read_text()).get("lane")
        except Exception:
            owner = None
        if owner not in live_lanes:
            c.unlink()
            n += 1
            print("released", k, "from", owner)
    print(f"{n} stale claims released")


def cmd_verify(a):
    """Durable-copy check: every done episode has a durable PASS status whose gates are all ok; frame/byte totals;
    re-hash a random sample of durable parquets against their status sha (light on /data3)."""
    ctl = CTL_ROOT / a.run
    wl, cfg = _load(ctl)
    dur = Path(cfg["durable_root"])
    done = sorted(os.listdir(ctl / "done"))
    out = {"run": a.run, "done_markers": len(done), "durable_status": 0, "missing_durable": [], "gate_fail": [],
           "frames": 0, "parquet_bytes": 0, "align_max": 0.0, "hosts": {}, "workers": {}}
    stats = []
    for k in done:
        split, ep = k.rsplit("_", 1)
        p = dur / split / "status" / f"episode_{ep}.json"
        if not p.exists():
            out["missing_durable"].append(k)
            continue
        st = json.loads(p.read_text())
        out["durable_status"] += 1
        g = st["gates"]
        ok = (st["status"] == "PASS" and g["events_ok"] and g["structure_ok"] and g["parquet_equals_live"]
              and g["contract"]["ok"] and g["contacts"]["ok"] and g["red_pixels"] == 0 and g["home"]["ok"]
              and g.get("alignment", {}).get("ok", False) and g["overridden_caller_actions"] == 0)
        if not ok:
            out["gate_fail"].append(k)
        out["frames"] += st["num_frames"]
        out["parquet_bytes"] += st["parquet_bytes"]
        out["align_max"] = max(out["align_max"], g["alignment"]["policy_max_abs_diff"])
        out["hosts"][st["host"]] = out["hosts"].get(st["host"], 0) + 1
        out["workers"][st["worker_sha256"][:12]] = out["workers"].get(st["worker_sha256"][:12], 0) + 1
        stats.append((split, ep, st))
    rnd = random.Random(0)
    sample = rnd.sample(stats, min(a.rehash, len(stats))) if stats else []
    bad = []
    for split, ep, st in sample:
        e = int(ep)
        f = dur / split / "robomme_data_lerobot/data" / f"chunk-{e // 1000:03d}" / f"episode_{e:06d}.parquet"
        if not f.exists() or sha256_file(f) != st["parquet_sha256"]:
            bad.append(f"{split}_{ep}")
    out["rehash_sampled"] = len(sample)
    out["rehash_mismatch"] = bad
    out["missing_durable_n"] = len(out["missing_durable"])
    out["missing_durable"] = out["missing_durable"][:20]
    out["pass"] = not out["gate_fail"] and not bad
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("run")
    p.add_argument("--out-root", required=True)
    p.add_argument("--durable-root", required=True)
    p.add_argument("--splits", default="train,dev48")
    p.add_argument("--episodes", default="")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--no-replay", action="store_true")
    p.add_argument("--negative-control", action="store_true")
    p.add_argument("--replay-obs-mode", default="state", help="replay env obs mode (state = physics only, no render)")
    p.add_argument("--home-ref", default="")
    p.add_argument("--tree-manifest", default="")
    p.add_argument("--lock-sha", default="")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("status")
    p.add_argument("run")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("retry-failed")
    p.add_argument("run")
    p.set_defaults(fn=cmd_retry_failed)
    p = sub.add_parser("release-stale")
    p.add_argument("run")
    p.add_argument("--minutes", type=int, default=20)
    p.set_defaults(fn=cmd_release_stale)
    p = sub.add_parser("verify")
    p.add_argument("run")
    p.add_argument("--rehash", type=int, default=10)
    p.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
