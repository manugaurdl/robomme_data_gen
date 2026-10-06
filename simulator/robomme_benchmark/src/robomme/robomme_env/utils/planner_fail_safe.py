"""Fail-aware wrappers for ManiSkill motion planners."""
from __future__ import annotations

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)


class ScrewPlanFailure(RuntimeError):
    """Raised when mplib reports a screw-planning failure."""


class _FailAwareMixin:
    """Mixin that turns ``-1`` screw-plan return values into exceptions."""

    def move_to_pose_with_screw(self, *args, **kwargs):  # type: ignore[override]
        result = super().move_to_pose_with_screw(*args, **kwargs)  # type: ignore[misc]
        if isinstance(result, int) and result == -1:
            raise ScrewPlanFailure("screw plan failed")
        return result

    def follow_path(self, result, refine_steps: int = 0):  # type: ignore[override]
        """BPP_v2: stop issuing path points as soon as the simulator owns control (accepted press -> automatic
        return) or the episode failed; the remaining planner commands are discarded. Unchanged when bpp_v2 is off."""
        base = getattr(self, "base_env", None)
        if not getattr(base, "bpp_v2", False):
            return super().follow_path(result, refine_steps=refine_steps)  # type: ignore[misc]
        import numpy as np

        n_step = result["position"].shape[0]
        total = n_step + refine_steps
        out = None
        for i in range(total):
            if base.bpp_v2_planner_must_stop():
                base.bpp_v2_record_planner_abort(total - i)
                break
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                action = np.hstack([qpos, result["velocity"][min(i, n_step - 1)]])
            else:
                action = np.hstack([qpos])
            out = self.env.step(action)
            self.elapsed_steps += 1
        return out


class FailAwarePandaArmMotionPlanningSolver(_FailAwareMixin, PandaArmMotionPlanningSolver):
    """Panda arm solver that raises on screw failures."""


class FailAwarePandaStickMotionPlanningSolver(
    _FailAwareMixin, PandaStickMotionPlanningSolver
):
    """Stick solver variant that raises on screw failures."""
