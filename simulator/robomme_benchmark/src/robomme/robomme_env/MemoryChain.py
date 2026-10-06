from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

#Robomme
from .utils import *
from .utils.subgoal_evaluate_func import static_check
from .utils import subgoal_language
from .utils.difficulty import normalize_robomme_difficulty

from ..logging_utils import logger


MEMORY_CHAIN_DOC_STRING = """**Task Description:**
A long-horizon, hybrid-memory task in a single persistent scene. The robot must:
(1) note which cube is highlighted (object-reference memory),
(2) watch that cube get hidden under one of several containers (spatial/permanence memory),
(3) perform an unrelated pick-and-place distractor loop N times (temporal/counting memory, also filler),
(4) press three buttons in a fixed order (procedural memory, also filler),
(5) press a "recall" button granting permission to retrieve,
(6) retrieve the container that hides the originally-highlighted cube's color, and
(7) place that cube onto the final target.

Stages 3-4 exist purely to create distance between the memory-write stages (1-2) and the
memory-read stage (6), so that success requires the agent to retain state across a much
longer horizon than any single existing RoboMME task suite exercises in isolation.
"""


@register_env("MemoryChain")
class MemoryChain(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    # difficulty controls: number of bins/distractors to choose among, number of
    # distractor pick-place repeats, and number of buttons in the procedural sequence
    config_hard = {
        "num_bins": 5,
        "distractor_repeats_min": 3,
        "distractor_repeats_max": 5,
        "num_procedure_buttons": 4,
    }
    config_easy = {
        "num_bins": 3,
        "distractor_repeats_min": 1,
        "distractor_repeats_max": 2,
        "num_procedure_buttons": 2,
    }
    config_medium = {
        "num_bins": 4,
        "distractor_repeats_min": 2,
        "distractor_repeats_max": 3,
        "num_procedure_buttons": 3,
    }
    configs = {
        "hard": config_hard,
        "easy": config_easy,
        "medium": config_medium,
    }

    COLOR_PALETTE = [
        {"color": (1, 0, 0, 1), "name": "red"},
        {"color": (0, 0, 1, 1), "name": "blue"},
        {"color": (0, 1, 0, 1), "name": "green"},
        {"color": (1, 1, 0, 1), "name": "yellow"},
        {"color": (1, 0, 1, 1), "name": "purple"},
    ]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0, seed=0,
                 Robomme_video_episode=None, Robomme_video_path=None, **kwargs):
        self.use_demonstrationwrapper = False
        self.demonstration_record_traj = False
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        self.robomme_failure_recovery = bool(kwargs.pop("robomme_failure_recovery", False))
        self.robomme_failure_recovery_mode = kwargs.pop("robomme_failure_recovery_mode", None)
        if isinstance(self.robomme_failure_recovery_mode, str):
            self.robomme_failure_recovery_mode = self.robomme_failure_recovery_mode.lower()

        self.seed = seed
        normalized_robomme_difficulty = normalize_robomme_difficulty(kwargs.pop("difficulty", None))
        if normalized_robomme_difficulty is not None:
            self.difficulty = normalized_robomme_difficulty
        else:
            seed_mod = seed % 3
            if seed_mod == 0:
                self.difficulty = "easy"
            elif seed_mod == 1:
                self.difficulty = "medium"
            else:
                self.difficulty = "hard"

        generator = torch.Generator()
        generator.manual_seed(seed)
        cfg_d = self.configs[self.difficulty]
        self.num_distractor_repeats = torch.randint(
            cfg_d["distractor_repeats_min"], cfg_d["distractor_repeats_max"] + 1, (1,), generator=generator
        ).item()

        self.procedure_achieved = []
        self.recall_unlocked = False
        self.highlight_starts = {}

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_eye = [0.3, 0, 0.4]
        camera_target = [0, 0, -0.2]
        pose = sapien_utils.look_at(eye=camera_eye, target=camera_target)
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=self.human_cam_eye_pos, target=self.human_cam_target_pos)
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        cfg_d = self.configs[self.difficulty]

        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # ------------------------------------------------------------------
        # Recall-permission button (stage 5)
        # ------------------------------------------------------------------
        recall_button_obb = build_button(
            self,
            center_xy=(-0.25, 0.2),
            scale=1.2,
            generator=generator,
            name="recall_button",
            randomize=False,
        )
        self.recall_button = self.button
        self.recall_button_joint = self.button_joint
        avoid = [recall_button_obb]

        # ------------------------------------------------------------------
        # Procedural-memory buttons (stage 4): pressed in a fixed order
        # ------------------------------------------------------------------
        self.procedure_buttons = []
        num_proc = cfg_d["num_procedure_buttons"]
        proc_positions = [(-0.2, -0.25 + 0.1 * i) for i in range(num_proc)]
        for i, pos in enumerate(proc_positions):
            obb = build_button(
                self,
                center_xy=pos,
                scale=0.9,
                generator=generator,
                name=f"procedure_button_{i}",
                randomize=False,
            )
            self.procedure_buttons.append(self.button)
            avoid.append(obb)
        order_perm = torch.randperm(num_proc, generator=generator).tolist()
        self.procedure_order = [self.procedure_buttons[i] for i in order_perm]

        # ------------------------------------------------------------------
        # Object-reference target cubes + spatial-memory bins (stages 1-2, 6)
        # ------------------------------------------------------------------
        num_bins = cfg_d["num_bins"]
        palette = self.COLOR_PALETTE[:num_bins]
        shuffle_indices = torch.randperm(len(palette), generator=generator).tolist()
        palette = [palette[i] for i in shuffle_indices]

        self.reference_cubes = []
        self.reference_cube_names = []
        self.reference_cube_colors = []
        for i, entry in enumerate(palette):
            cube = spawn_random_cube(
                self,
                color=entry["color"],
                avoid=avoid,
                include_existing=False,
                include_goal=False,
                region_center=[0.15, 0],
                region_half_size=0.15,
                half_size=self.cube_half_size,
                min_gap=self.cube_half_size * 2,
                random_yaw=True,
                name_prefix=f"ref_cube_{entry['name']}",
                generator=generator,
            )
            self.reference_cubes.append(cube)
            self.reference_cube_names.append(f"ref_cube_{entry['name']}")
            self.reference_cube_colors.append(entry["name"])
            setattr(self, f"ref_cube_{entry['name']}", cube)
            avoid.append(cube)

        target_idx = torch.randint(0, len(self.reference_cubes), (1,), generator=generator).item()
        self.target_cube = self.reference_cubes[target_idx]
        self.target_cube_color = self.reference_cube_colors[target_idx]
        self.non_target_cubes = [c for c in self.reference_cubes if c is not self.target_cube]

        self.spawned_bins = []
        for i in range(num_bins):
            bin_actor = spawn_random_bin(
                self,
                avoid=avoid,
                region_center=[-0.05, 0.15],
                region_half_size=0.15,
                min_gap=self.cube_half_size * 2,
                name_prefix=f"bin_{i}",
                max_trials=256,
                generator=generator,
            )
            self.spawned_bins.append(bin_actor)
            setattr(self, f"bin_{i}", bin_actor)
            avoid.append(bin_actor)

        hide_idx = torch.randint(0, len(self.spawned_bins), (1,), generator=generator).item()
        self.hide_bin = self.spawned_bins[hide_idx]
        self.non_hide_bins = [b for b in self.spawned_bins if b is not self.hide_bin]

        # ------------------------------------------------------------------
        # Distractor cube for stages 3 (temporal/counting filler)
        # ------------------------------------------------------------------
        self.distractor_cube = spawn_random_cube(
            self,
            color=(0.5, 0.5, 0.5, 1),
            avoid=avoid,
            include_existing=False,
            include_goal=False,
            region_center=[0.15, -0.2],
            region_half_size=0.08,
            half_size=self.cube_half_size,
            min_gap=self.cube_half_size,
            random_yaw=True,
            name_prefix="distractor_cube",
            generator=generator,
        )
        avoid.append(self.distractor_cube)
        self.distractor_target = spawn_random_target(
            self,
            avoid=avoid,
            include_existing=False,
            include_goal=False,
            region_center=[0.15, -0.2],
            region_half_size=0.08,
            radius=self.cube_half_size * 2,
            thickness=0.005,
            min_gap=self.cube_half_size * 2,
            name_prefix="distractor_target",
            generator=generator,
        )
        avoid.append(self.distractor_target)

        # Final retrieval target (stage 7)
        self.final_target = spawn_random_target(
            self,
            avoid=avoid,
            include_existing=False,
            include_goal=False,
            region_center=[0.15, 0.2],
            region_half_size=0.08,
            radius=self.cube_half_size * 2,
            thickness=0.005,
            min_gap=self.cube_half_size * 2,
            name_prefix="final_target",
            generator=generator,
        )

        self.task_list = self._build_task_list()

        self.recovery_pickup_indices, self.recovery_pickup_tasks = task4recovery(self.task_list)
        if self.robomme_failure_recovery:
            self.fail_grasp_task_index = inject_fail_grasp(
                self.task_list, generator=generator, mode=self.robomme_failure_recovery_mode,
            )
        else:
            self.fail_grasp_task_index = None

    def _build_task_list(self):
        tasks = []

        # ---- Stage 1: object-reference — cube is highlighted, no action required ----
        tasks.append({
            "func": lambda: before_absTimestep(self, absTimestep=100),
            "name": "observe the highlighted cube",
            "subgoal_segment": "observe the highlighted cube at <>",
            "choice_label": "observe the highlighted cube",
            "demonstration": True,
            "failure_func": None,
            "solve": lambda env, planner: solve_hold_obj_absTimestep(env, planner, absTimestep=100),
            "segment": self.target_cube,
        })

        # ---- Stage 2: spatial/permanence — watch the target cube get hidden ----
        tasks.append({
            "func": lambda: self._cube_is_hidden(),
            "name": "watch the cube get hidden under a container",
            "subgoal_segment": "watch the cube get hidden under the container at <>",
            "choice_label": "watch the cube get hidden",
            "demonstration": True,
            "failure_func": None,
            "solve": lambda env, planner: self._solve_hide_cube(env, planner),
            "segment": self.hide_bin,
        })

        # ---- Stage 3: temporal/counting filler — N pickup/drop cycles on a distractor ----
        for i in range(self.num_distractor_repeats):
            tasks.append({
                "func": (lambda: is_obj_pickup(self, obj=self.distractor_cube)),
                "name": subgoal_language.get_subgoal_with_index(
                    i, "pick up the distractor cube for the {idx} time"
                ),
                "subgoal_segment": subgoal_language.get_subgoal_with_index(
                    i, "pick up the distractor cube at <> for the {idx} time"
                ),
                "choice_label": "pick up the distractor cube",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self, self.spawned_bins),
                "solve": lambda env, planner: solve_pickup(env, planner, obj=self.distractor_cube),
                "segment": self.distractor_cube,
            })
            tasks.append({
                "func": (lambda: is_obj_dropped_onto(self, obj=self.distractor_cube, target=self.distractor_target)),
                "name": "place the distractor cube onto its target",
                "subgoal_segment": "place the distractor cube onto the target at <>",
                "choice_label": "place the distractor cube onto the target",
                "demonstration": False,
                "failure_func": lambda: is_any_bin_pickup(self, self.spawned_bins),
                "solve": lambda env, planner: solve_putonto_whenhold(env, planner, target=self.distractor_target),
                "segment": self.distractor_target,
            })

        # ---- Stage 4: procedural memory filler — press buttons in fixed order ----
        for i, button in enumerate(self.procedure_order):
            tasks.append({
                "func": (lambda b=button: self._button_pressed_in_order(b)),
                "name": subgoal_language.get_subgoal_with_index(i, "press the {idx} button in the sequence"),
                "subgoal_segment": subgoal_language.get_subgoal_with_index(
                    i, "press the {idx} button in the sequence at <>"
                ),
                "choice_label": "press the button",
                "demonstration": False,
                "failure_func": lambda b=button: self._wrong_procedure_button(expected=b),
                "solve": lambda env, planner, b=button: solve_button(env, planner, obj=b),
                "segment": None,
            })

        # ---- Stage 5: press recall button, granting retrieval permission ----
        tasks.append({
            "func": lambda: is_button_pressed(self, obj=self.recall_button),
            "name": "press the recall button",
            "subgoal_segment": "press the recall button at <>",
            "choice_label": "press the recall button",
            "demonstration": False,
            "failure_func": lambda: is_any_bin_pickup(self, self.spawned_bins),
            "solve": lambda env, planner: solve_button(env, planner, obj=self.recall_button),
            "segment": None,
        })

        # ---- Stage 6: retrieve — pick up the bin hiding the remembered cube ----
        tasks.append({
            "func": (lambda: is_bin_pickup(self, obj=self.hide_bin)),
            "name": f"pick up the container hiding the {self.target_cube_color} cube",
            "subgoal_segment": f"pick up the container at <> hiding the {self.target_cube_color} cube",
            "choice_label": "pick up the container",
            "demonstration": False,
            "failure_func": lambda: is_any_bin_pickup(self, self.non_hide_bins),
            "solve": lambda env, planner: solve_pickup_bin(env, planner, obj=self.hide_bin),
            "segment": self.hide_bin,
        })
        tasks.append({
            "func": (lambda: is_bin_putdown(self, obj=self.hide_bin)),
            "name": "put down the container",
            "subgoal_segment": "put down the container",
            "choice_label": "put down the container",
            "demonstration": False,
            "failure_func": lambda: is_any_bin_pickup(self, self.non_hide_bins),
            "solve": lambda env, planner: solve_putdown_whenhold(env, planner),
            "segment": None,
        })

        # ---- Stage 7: pick up the recalled cube and place it on the final target ----
        tasks.append({
            "func": (lambda: is_obj_pickup(self, obj=self.target_cube)),
            "name": f"pick up the {self.target_cube_color} cube",
            "subgoal_segment": f"pick up the {self.target_cube_color} cube at <>",
            "choice_label": "pick up the cube",
            "demonstration": False,
            "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
            "solve": lambda env, planner: solve_pickup(env, planner, obj=self.target_cube),
            "segment": self.target_cube,
        })
        tasks.append({
            "func": (lambda: is_obj_dropped_onto(self, obj=self.target_cube, target=self.final_target)),
            "name": "place the recalled cube onto the final target",
            "subgoal_segment": "place the recalled cube onto the final target at <>",
            "choice_label": "place the cube onto the target",
            "demonstration": False,
            "failure_func": lambda: is_any_obj_pickup(self, self.non_target_cubes),
            "solve": lambda env, planner: solve_putonto_whenhold(env, planner, target=self.final_target),
            "segment": self.final_target,
        })

        return tasks

    # ----------------------------------------------------------------------
    # Task-specific helpers
    # ----------------------------------------------------------------------
    def _cube_is_hidden(self):
        # "Hidden" once the target cube is no longer visible above the table
        # (moved under the hide_bin footprint and lowered) and the demo step
        # window for this stage has elapsed.
        cube_pos = self.target_cube.pose.p[0]
        bin_pos = self.hide_bin.pose.p[0]
        horizontal_distance = torch.sqrt(
            (cube_pos[0] - bin_pos[0]) ** 2 + (cube_pos[1] - bin_pos[1]) ** 2
        )
        return bool(horizontal_distance <= self.cube_half_size * 3) and before_absTimestep(self, absTimestep=200)

    def _solve_hide_cube(self, env, planner):
        # Scripted (non-physical) relocation used only for oracle demonstration
        # trajectories: teleport the target cube under the hide_bin footprint.
        bin_pos = self.hide_bin.pose.p[0].tolist()
        with torch.no_grad():
            self.target_cube.set_pose(
                sapien.Pose(p=[bin_pos[0], bin_pos[1], self.cube_half_size / 2])
            )
        return solve_hold_obj_absTimestep(env, planner, absTimestep=200)

    def _button_pressed_in_order(self, button):
        if is_button_pressed(self, obj=button):
            if button not in self.procedure_achieved:
                self.procedure_achieved.append(button)
            return True
        return False

    def _wrong_procedure_button(self, expected):
        for button in self.procedure_buttons:
            if button is expected or button in self.procedure_achieved:
                continue
            if is_button_pressed(self, obj=button):
                return True
        return False

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            qpos = reset_panda.get_reset_panda_param("qpos")
            self.agent.reset(qpos)
            self.procedure_achieved = []

    def _get_obs_extra(self, info: Dict):
        return dict()

    def evaluate(self, solve_complete_eval=False):
        previous_failure = getattr(self, "failureflag", None)
        self.successflag = torch.tensor([False])
        if previous_failure is not None and bool(previous_failure.item()):
            self.failureflag = previous_failure
        else:
            self.failureflag = torch.tensor([False])

        if self.use_demonstrationwrapper == False:
            allow_subgoal_change_this_timestep = bool(solve_complete_eval)
        else:
            allow_subgoal_change_this_timestep = bool(solve_complete_eval or not self.demonstration_record_traj)

        all_tasks_completed, current_task_name, task_failed, self.current_task_specialflag = sequential_task_check(
            self, self.task_list, allow_subgoal_change_this_timestep=allow_subgoal_change_this_timestep
        )

        if task_failed:
            self.failureflag = torch.tensor([True])
            logger.debug(f"Task failed: {current_task_name}")

        if all_tasks_completed and not task_failed:
            self.successflag = torch.tensor([True])

        return {
            "success": self.successflag,
            "fail": self.failureflag,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(self.agent.tcp_pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        return reaching_reward * 0

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5

    #Robomme
    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        timestep = self.elapsed_steps
        highlight_obj(self, self.target_cube, start_step=0, end_step=100, cur_step=timestep)
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
