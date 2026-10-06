"""BPP_v2 PatternLock mechanics: automatic return home after every accepted button press.

Opt-in via ``gym.make("PatternLock", ..., bpp_v2=True)`` (default off; the v1/050f770 behaviour is untouched when off).
One implementation is shared by the recorder (RecordWrapper + planner) and the evaluator (DemonstrationWrapper).

Episode (password pw[0..m]):
  row 0            home reset frame (sim)
  demo             for i in 0..m: demo_reach home->pw[i] (expert planner) | return (20 rows, sim) | home (1 row, sim)
  handoff          handoff_reach home->pw[0] (sim-owned planner reach, never a policy target) | return | home
  execution        for i in 1..m: exec_reach home->pw[i] (policy; the expert planner stands in when recording) | return | home
  success          declared on the terminal home row (after the final press's return + snap)

Row convention (unchanged from v1): row t = observation rendered after env.step t, action[t] = joint targets applied at
step t, state[t] = qpos after step t. The contact row (the step whose evaluate() accepts a press) keeps its reach action;
the return starts at the next step. Return = 20 joint-space linear-interpolation targets contact_qpos -> HOME
(k/20, k=1..20) through the pd_joint_pos controller. Home row = one more step with target HOME, after which the
simulator snaps qpos=HOME, qvel=0, qf=0, drive target=HOME, controller state=HOME and re-renders; that re-rendered
observation is the home row (the settled home decision frame). Task progress, contacts and failure checks are frozen on
return/home steps. Contacts are processed only on reach steps; an accepted press latches (the phase leaves "reach" inside
the same evaluate() call), so repeated evaluate() calls cannot add progress. Wrong button -> immediate failure (v1 rule,
incl. the previous-button debounce exemption, which neither progresses, fails nor triggers a return).

This module is simulator-free so the timing/segment contract can be unit-tested without SAPIEN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

BPP_V2_VERSION = "bpp_v2_mechanics_1"
RETURN_STEPS = 20

# D2 home: ONE fixed joint config. TCP (-0.1, 0.0, 0.14) world = above the button-grid centre (buttons at z=0.01; 4 cm
# above the z<0.1 contact slab), v1 home TCP orientation (wxyz ~ (0,1,0,0): every reach keeps the current orientation),
# SAPIEN-pinocchio IK seeded at the median BPP_v1 dev48 joint state. z=0.14 is the highest 1-cm TCP height whose solution
# lies inside the frozen v1 action min-max range (stats sha b36c0091...): normalized [-0.012,-0.684,-0.032,-0.048,
# -0.077,-0.982,0.001]; z=0.15/0.16 put q6 at -1.15/-1.33. Report: .bpp_v2/reports/home_pose.json.
HOME_QPOS = np.asarray(
    [-0.00147877566, 0.110702187, 0.00146709458, -2.33247628, -0.000253323766, 2.44317490, 0.785576286],
    dtype=np.float32,
)
HOME_TCP_WORLD = (-0.1, 0.0, 0.14)

OWNER_POLICY = "policy"
OWNER_SIM = "sim"
KIND_HOME = "home"
KIND_RETURN = "return"
KIND_DEMO_REACH = "demo_reach"
KIND_HANDOFF_REACH = "handoff_reach"
KIND_EXEC_REACH = "exec_reach"
REACH_KINDS = (KIND_DEMO_REACH, KIND_HANDOFF_REACH, KIND_EXEC_REACH)

PHASE_REACH = "reach"    # caller (policy / expert planner / sim handoff planner) owns the next step
PHASE_RETURN = "return"  # next step is a sim-owned return interpolation step
PHASE_HOME = "home"      # next step is the sim-owned home step (snap + re-render)
PHASE_DONE = "done"      # terminal home row reached (success)


def return_targets(contact_qpos, home_qpos=None, steps: int = RETURN_STEPS) -> np.ndarray:
    """Joint-space linear interpolation contact -> home, k/steps for k = 1..steps (last row == home exactly)."""
    home = HOME_QPOS if home_qpos is None else np.asarray(home_qpos, dtype=np.float32)
    start = np.asarray(contact_qpos, dtype=np.float64).reshape(-1)[:7]
    alphas = np.arange(1, steps + 1, dtype=np.float64)[:, None] / float(steps)
    targets = start[None, :] + alphas * (home.astype(np.float64)[None, :] - start[None, :])
    targets = targets.astype(np.float32)
    targets[-1] = home  # exact
    return targets


def owner_of(kind: str) -> str:
    """Owner of a row's action, T0 trainer-contract semantics (.bpp_v2/agents/T0_train_prep.md §3.4): "policy" =
    policy/expert-owned (demo_reach rows = the demonstrator's reaches, exec_reach rows = the policy's; the expert planner
    stands in when recording), "sim" = row 0, the sim's pw[0] handoff reach, every return row, every settled-home row."""
    return OWNER_POLICY if kind in (KIND_DEMO_REACH, KIND_EXEC_REACH) else OWNER_SIM


@dataclass
class BPPV2State:
    """Per-episode auto-return state machine + row metadata (one row per env.step, row 0 = reset frame)."""

    phase: str = PHASE_REACH
    step_kind: Optional[str] = None       # phase of the env.step currently executing (None outside steps)
    return_k: int = 0
    targets: Optional[np.ndarray] = None
    terminal_pending: bool = False        # the last accepted press completed the password (success at its home row)
    success: bool = False
    failed: bool = False
    row: int = 0
    segment_id: int = -1                  # T0 contract: -1 on sim-owned rows, 0,1,2,... per contiguous policy/expert reach
    segment_kind: str = KIND_HOME
    next_segment_id: int = 0
    exec_started: bool = False            # True from the settled-home row after the handoff (= exec_start_idx) on
    last_accept_kind: Optional[str] = None
    accepted_this_step: int = -1
    events: List[dict] = field(default_factory=list)
    planner_aborts: List[dict] = field(default_factory=list)
    overridden_caller_actions: int = 0    # sim-owned steps whose caller passed a different action (discarded)
    last_row: dict = field(default_factory=dict)

    def reset(self, home_qpos) -> dict:
        self.__init__()
        self.last_row = self._row_meta(KIND_HOME, OWNER_SIM, home_qpos, accepted=-1)
        return self.last_row

    # -- transitions -------------------------------------------------------------------------------------------------
    def begin_step(self, reach_kind: Optional[str]):
        """Return (row_kind, owner) for the env.step about to run, given the kind of the current reach task."""
        self.step_kind = self.phase
        self.accepted_this_step = -1
        if self.phase == PHASE_REACH:
            kind = reach_kind if reach_kind in REACH_KINDS else KIND_EXEC_REACH
        elif self.phase == PHASE_RETURN:
            kind = KIND_RETURN
        else:  # home or done: hold/snap home
            kind = KIND_HOME
        return kind, owner_of(kind)

    def scheduled_action(self, home_qpos) -> Optional[np.ndarray]:
        if self.phase == PHASE_RETURN:
            return np.asarray(self.targets[self.return_k], dtype=np.float32)
        if self.phase in (PHASE_HOME, PHASE_DONE):
            return np.asarray(home_qpos, dtype=np.float32)
        return None

    def accept(self, button_index: int, task_index: int, reach_kind: str, contact_qpos, home_qpos, completes: bool,
               elapsed_step: int):
        """Latch an accepted press: the next step starts the return. Called from evaluate() on a reach step."""
        self.targets = return_targets(contact_qpos, home_qpos)
        self.return_k = 0
        self.phase = PHASE_RETURN
        self.accepted_this_step = int(button_index)
        self.terminal_pending = bool(completes)
        self.last_accept_kind = reach_kind
        self.events.append({
            "row": int(self.row + 1 if self.step_kind is not None else self.row),
            "elapsed_step": int(elapsed_step),
            "button": int(button_index),
            "task_index": int(task_index),
            "reach_kind": reach_kind,
            "completes_password": bool(completes),
            "contact_qpos": [float(x) for x in np.asarray(contact_qpos).reshape(-1)[:7]],
        })

    def end_step(self, kind: str, owner: str, applied_action) -> dict:
        """Advance counters after the env.step; returns the new row's metadata."""
        stepped = self.step_kind
        if stepped == PHASE_RETURN:
            self.return_k += 1
            if self.return_k >= len(self.targets):
                self.phase = PHASE_HOME
        elif stepped in (PHASE_HOME, PHASE_DONE):
            if self.last_accept_kind == KIND_HANDOFF_REACH:
                self.exec_started = True  # this settled-home row is the first execution decision frame (exec_start)
            if self.terminal_pending:
                self.phase = PHASE_DONE
                self.success = True
            else:
                self.phase = PHASE_REACH
        self.row += 1
        if owner == OWNER_POLICY:
            if kind != self.segment_kind or self.segment_id < 0:
                self.segment_id = self.next_segment_id
                self.next_segment_id += 1
        else:
            self.segment_id = -1
        self.segment_kind = kind
        self.last_row = self._row_meta(kind, owner, applied_action, accepted=self.accepted_this_step)
        self.step_kind = None
        return self.last_row

    def sim_owns_next_step(self) -> bool:
        return (not self.failed) and self.phase in (PHASE_RETURN, PHASE_HOME)

    def planner_must_stop(self) -> bool:
        return self.failed or self.phase != PHASE_REACH

    def _row_meta(self, kind: str, owner: str, applied_action, accepted: int) -> dict:
        return {
            "row": int(self.row),
            "control_owner": owner,
            "segment_id": int(self.segment_id),
            "segment_kind": kind,
            "accepted_event": int(accepted),
            "is_demo": not self.exec_started,
            "applied_action": np.asarray(applied_action, dtype=np.float32).reshape(-1)[:7].copy(),
            "phase_after": self.phase,
        }
