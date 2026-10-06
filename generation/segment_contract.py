"""Standalone BPP_v2 reach-boundary labels for generated episodes."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ACTION_PRED_HORIZON = 16


def segment_anchor_expectations(
    arrays: Mapping[str, np.ndarray], *, episode: int, num_frames: int,
    exec_start: int, transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select execution decisions and their within-reach action targets."""
    owner = np.asarray(arrays["control_owner"])
    segment = np.asarray(arrays["segment_id"])
    if owner.shape != (num_frames,) or segment.shape != (num_frames,):
        raise ValueError(f"episode {episode}: invalid owner/segment shape")
    if owner.dtype.kind not in "biu" or segment.dtype.kind not in "iu":
        raise ValueError(f"episode {episode}: owner/segment must be integers")
    owner = owner.astype(np.int64)
    segment = segment.astype(np.int64)
    if np.any((owner != 0) & (owner != 1)):
        raise ValueError(f"episode {episode}: owner must be 0 (sim) or 1 (policy)")
    if np.any(segment[owner == 0] != -1) or np.any(segment[owner == 1] < 0):
        raise ValueError(f"episode {episode}: segment_id must be -1 exactly on sim rows")
    if not 0 <= exec_start < num_frames:
        raise ValueError(f"episode {episode}: invalid execution start")
    transition = np.asarray(transition, dtype=bool)
    if transition.shape != (num_frames,) or np.any(transition & (owner == 1)):
        raise ValueError(f"episode {episode}: counter may rise only on sim rows")

    segment_end: dict[int, int] = {}
    for value in np.unique(segment[owner == 1]):
        rows = np.flatnonzero(segment == value)
        if int(rows[-1] - rows[0] + 1) != rows.size:
            raise ValueError(f"episode {episode}: segment {value} is not contiguous")
        if rows[0] <= exec_start < rows[-1]:
            raise ValueError(f"episode {episode}: segment {value} straddles execution start")
        segment_end[int(value)] = int(rows[-1])

    valid = np.zeros(num_frames, dtype=bool)
    targets = np.full((num_frames, ACTION_PRED_HORIZON), -1, dtype=np.int64)
    padded = np.zeros((num_frames, ACTION_PRED_HORIZON), dtype=bool)
    for frame in range(exec_start, num_frames - 1):
        if owner[frame + 1] != 1:
            continue
        end = segment_end[int(segment[frame + 1])]
        raw = frame + 1 + np.arange(ACTION_PRED_HORIZON, dtype=np.int64)
        valid[frame] = True
        targets[frame] = np.minimum(raw, end)
        padded[frame] = raw > end
    if not valid.any():
        raise ValueError(f"episode {episode}: no execution decisions")
    return valid, targets, padded
