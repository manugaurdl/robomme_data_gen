# Codex handoff: generate BPP_v2 PatternLock data

You have a standalone **data-generation source bundle**. Build and run a generator for RoboMME PatternLock Easy using these files. Work inside this bundle and your own output directory.

## Task and fixed simulator contract

Generate the task shown in “Patternlock: BPP with resets”: a 3×3 button grid, two RGB cameras, a demonstration and simulator-owned first-button handoff, then policy-owned button reaches. There are **no red flashes, persistent highlights, or movement trails**. After each accepted press, the **simulator** performs 20 return steps and one settled-home step; the next reach starts from the identical home pose. A wrong button fails. Keep `bpp_v2=True` for both recording and any replay. “Manual resets” is not the mechanism in this version.

The copied simulator lives in `simulator/`. Its `robomme_benchmark/src/robomme/` package includes shared RoboMME task modules required by its imports. The two copied scripts under `simulator/memory_diffusion_policy/scripts/` handle episode export and Parquet conversion. The raw worker and selection coordinator are in `generation/`. `generation/segment_contract.py` labels decision frames and keeps action targets within each reach. `generation/p5_build.py` is the optional decoded-cache, counter, and label builder. `simulator/venv_sim.freeze.txt` records the generation environment. The RoboMME Apache-2.0 license is included.

This is a **source handoff**, not a preinstalled environment or a finished dataset. The source was copied from the BPP_v2 generation snapshot. The simulator subset has its own content lock; check it with `python simulator/bpp_v2_simulator.py check simulator`. Do not compare this trimmed bundle's tree digest to the larger original tree digest. The raw generator was smoke-tested on Trinity on 2026-10-06; the optional `p5_build.py` postprocessor was not tested and still needs LeRobot metadata templates and adaptation to this bundle's simulator lock.

## Choices to get from me before the full run

| Choice | Recommended starting point | Why it matters |
| --- | --- | --- |
| Episode identities | **New deterministic seeds and passwords** | Reproducing the original 5,000 episode identities requires the original *data selection manifest* and source route variants. |
| Splits and counts | First 1 episode, then a 24-episode pilot; choose final train/dev/test counts afterward | The presentation used 5,000 train and 48 development episodes. A fresh test split is optional. Keep ordered full passwords disjoint across splits. |
| Password rule | PatternLock **Easy**, full password length 2–4, execution length 1–3, no repeated buttons | Longer or repeated passwords belong to a different simulator/data contract. A 5,000-episode set cannot be assumed to have 5,000 distinct Easy passwords; use documented route variants if more samples are wanted. |
| Deliverable | **Raw Parquets + status/trace/rowlog first**; add decoded arrays, counters, labels, and stats only if I need training-ready data | Raw data are easier to validate and store. The optional cache can be much larger. |
| Compute and storage | A Linux NVIDIA GPU with Vulkan/SAPIEN; specify output and durable storage paths outside this source bundle | The renderer cannot be validated on a CPU-only laptop. The original 5,000-episode raw Parquets were about 254 GB; decoded images added about 673 GB. |

If I do not answer a choice, run only the one-episode smoke test and report what is missing before a large run. Do not silently change the simulator mechanics to make generation easier.

## Build the generator

1. Set up Python 3.11 and the pinned simulator stack: ManiSkill fork `YinpeiDai/ManiSkill@07be6fbc66350ddca200abfb0a11b692f078f7fd`, SAPIEN 3.0.3, Torch 2.9.1+cu128, and the other generation packages in `simulator/venv_sim.freeze.txt`. Use the bundle's `robomme_benchmark/src` on `PYTHONPATH`. The copied `uv.lock` alone resolved SAPIEN 3.0.2 in the original campaign, so verify installed versions and imports rather than assuming `uv sync --frozen` reproduces the recorded environment. Run rendering on a compute GPU, not a login node.
2. Make a deterministic selection manifest for each chosen split. `generation/make_selection.py` creates fresh disjoint Easy passwords: for example, `python generation/make_selection.py --out-root inputs/selection --train 2 --dev48 1 --seed-start 15001`. `generation/gen_ctl.py` expects `train/selection/manifest.json` and `dev48/selection/manifest.json` under `BPP_SELECTION_ROOT` (default `inputs/selection/`). Records contain episode ID, seed, ordered `full_password`, and `source`; route variants need their route segments and base-episode identity. The helper selects *unique* ordered passwords and intentionally stops if the requested count is impossible. For large sets, implement documented route variants rather than accepting silent duplicates. Validate every record against the simulator and prove split disjointness before recording.
3. Make a LeRobot Parquet **schema template** with `python generation/make_template.py --out inputs/template.parquet`; this is the default `BPP_TEMPLATE_PARQUET`. `write_patternlock_episode_parquet.py` adds the BPP_v2 columns `control_owner`, `segment_id`, `segment_kind`, and `accepted_event`. Verify a real episode round-trips through the fresh template. The original BPP_v1 template was not included.
4. Adapt the copied `gen_ctl.py` and `gen_worker.py` to your host's paths and process supervision. The worker already resolves `simulator/` relative to this bundle and uses `segment_contract.py`. The coordinator accepts `BPP_SELECTION_ROOT`, `BPP_TEMPLATE_PARQUET`, and `BPP_GEN_CONTROL_ROOT`. It writes a control directory for the worker. Put generated data under a separate, versioned output root. Configure multi-node launch commands for your own cluster, nodes, and storage.
5. Record **one** seeded episode, convert HDF5 → NPZ → Parquet, and inspect its PASS status and replay. The worker checks source hashes, row order, seed/password agreement, contact events, 20-return-plus-home timing, frame/action alignment, and physics replay. Check that the first execution frame is settled home; counter changes on the first return row; every action target remains in its policy-owned reach; no visual progress cue appears. Then run a pilot across representative password lengths and route variants before scaling.
6. If I request training-ready data, port `generation/p5_build.py` to your output paths and provide its LeRobot `info.json` and `tasks.jsonl` templates under `generation/templates/`. It builds decoded images, states/actions, accepted-event counters, row labels, statistics, and manifests from passed raw episodes. It uses the included segment rule. Validate outputs with independent data and replay checks. Keep raw episodes and checksums so caches can be rebuilt.

Do not skip the seed/password check, no-flash check, return/home check, or observation/action replay. Save the selection, code/runtime hashes, per-episode status, and a concise README beside the generated data. Do not run a long campaign until the single-episode and representative pilot checks pass.

## Small Trinity smoke test already run

On 2026-10-06, this portable bundle was staged under `/scratch/mgaur/robomme_data_gen_verify_20261006_codex/bundle` on `trinity-2-3` and run with the existing Python 3.11 simulator environment on one idle RTX 3090. The inputs, control files, and outputs were placed under that separate scratch directory. The source repository and existing BPP datasets were not changed. These are the key commands; `python` means the simulator environment's Python, and `WORK` is an output directory outside the bundle:

```sh
WORK=/scratch/mgaur/robomme_data_gen_verify_20261006_codex
export BPP_SELECTION_ROOT="$WORK/selection"
export BPP_TEMPLATE_PARQUET="$WORK/template.parquet"
export BPP_GEN_CONTROL_ROOT="$WORK/control"
python generation/make_selection.py --out-root "$BPP_SELECTION_ROOT" --train 1 --dev48 1 --seed-start 15001
python generation/make_template.py --out "$BPP_TEMPLATE_PARQUET"
python generation/gen_ctl.py plan smoke --out-root "$WORK/raw" --durable-root "$WORK/raw" --pilot --negative-control
CUDA_VISIBLE_DEVICES=0 python generation/gen_worker.py --ctl "$WORK/control/smoke" --lane smoke0 --max-episodes 2
python generation/gen_ctl.py verify smoke --rehash 2
python generation/gate_pilot.py "$WORK/raw" "$WORK/control/smoke" "$WORK/gate_pilot.json"
CUDA_VISIBLE_DEVICES=0 python generation/replay_pure.py "$WORK/raw" "$WORK/control/smoke" "$WORK/replay_pure.json"
```

The worker's exit code `3` here means its requested two-episode cap was reached. Both episodes had PASS statuses: 496 frames total, 44,254,296 Parquet bytes. The independent gate found zero red pixels across both cameras, identical settled-home frames, no wrong-button contacts, correct alignment, and successful negative-control divergence. The fresh-env replay matched all 496 robot states and both camera images exactly, with both episodes successful. This confirms the **raw generation path for two new Easy passwords**; it is not a full-campaign or optional postprocessing validation.

## Boundaries of this bundle

It contains simulator source, generation/export/conversion code, a data-label rule, optional data postprocessing, and this handoff. The default selection helper generates new episodes with the same simulator mechanics. Matching the original episode identities requires the original selection inputs.
