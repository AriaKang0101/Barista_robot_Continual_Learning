"""Identical goal-conditioned environment for both learning methods."""
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import goal_errors, reached, validate_tasks


class BaristaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, simulator, tasks, split="train", observation_mode="state_goal"):
        super().__init__()
        validate_tasks(tasks)
        if split not in ("train", "eval"):
            raise ValueError("split must be train or eval")
        self.sim = simulator
        self.specification = tasks
        self.tasks = {t["id"]: t for t in tasks["tasks"]}
        self.schema = tasks["schema_version"]
        if split == "eval":
            if self.schema == 4:
                self.starts = {k: np.asarray(v, dtype=float) for k, v in tasks["eval_starts"].items()}
                self.eval_bands = tasks["eval_bands"]
            else:
                self.starts = np.asarray(tasks["eval_starts"], dtype=float)
                self.eval_bands = None
        else:
            self.starts, self.eval_bands = None, None
        self.split = split
        if observation_mode not in ("state_goal", "image_state_goal", "image_goal", "image_goal_joints"):
            raise ValueError("Invalid observation_mode")
        self.observation_mode = observation_mode
        self.tolerance = tasks["goal_tolerance"]
        self.max_steps = tasks["max_steps"]
        sampler = tasks["training_sampler"]
        self.joint_lower = np.asarray(sampler["lower"], dtype=np.float32)
        self.joint_upper = np.asarray(sampler["upper"], dtype=np.float32)
        self.curriculum_fraction = 1.0
        self.task = "A"
        self.frames = deque(maxlen=4)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        observations = {}
        if observation_mode in ("state_goal", "image_state_goal"):
            # Current q, current dq, target q. This is a compact Markov state
            # and makes the first CL benchmark about retention, not perception.
            observations["state"] = spaces.Box(-1.0, 1.0, (6,), np.float32)
        if observation_mode in ("image_state_goal", "image_goal", "image_goal_joints"):
            observations["image"] = spaces.Box(0, 255, (4, 84, 84), np.uint8)
        if observation_mode in ("image_goal", "image_goal_joints"):
            observations["goal"] = spaces.Box(-np.inf, np.inf, (6,), np.float32)
        if observation_mode == "image_goal_joints":
            observations["joints"] = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Dict(observations)
        self.base_position = np.asarray(tasks["fixed_geometry"]["base"]).reshape(3, 4)[:, 3]
        self.sim.assert_geometry(tasks["fixed_geometry"])
        if self.sim.runtime != tasks["runtime"]:
            raise RuntimeError("Simulator version/engine/timestep differs from preparation. Prepare tasks again.")
        self.done = True

    def set_task(self, task):
        if task not in self.tasks:
            raise ValueError(f"Unknown task {task}")
        self.task = task
        self.done = True

    def set_curriculum_fraction(self, fraction):
        if self.split != "train" or not 0 < fraction <= 1:
            raise ValueError("Training curriculum fraction must be in (0, 1]")
        self.curriculum_fraction = float(fraction)

    @property
    def goal(self):
        return np.asarray(self.tasks[self.task]["points"])

    def observation(self):
        q = self.sim.joint_positions().astype(np.float32)
        normalized_q = 2.0 * (q - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0
        target_q = np.asarray(self.tasks[self.task]["q"], dtype=np.float32)
        normalized_target = 2.0 * (target_q - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0
        observation = {}
        if self.observation_mode in ("state_goal", "image_state_goal"):
            velocity = np.clip(self.sim.joint_velocities().astype(np.float32) / 0.8, -1.0, 1.0)
            observation["state"] = np.concatenate([
                np.clip(normalized_q, -1.0, 1.0), velocity,
                np.clip(normalized_target, -1.0, 1.0)]).astype(np.float32)
        if self.observation_mode in ("image_state_goal", "image_goal", "image_goal_joints"):
            observation["image"] = np.stack(self.frames)
        if self.observation_mode in ("image_goal", "image_goal_joints"):
            observation["goal"] = (self.goal - self.base_position).astype(np.float32).ravel()
        if self.observation_mode == "image_goal_joints":
            observation["joints"] = np.clip(normalized_q, -1.0, 1.0).astype(np.float32)
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        index = options.get("start_index")
        if self.split == "train":
            if index is not None:
                raise ValueError("Training uses continuous safe random starts, not start_index")
            goals = [t["points"] for t in self.tasks.values()]
            anchor_indices = None
            if self.schema == 4:
                order = self.specification["training_sampler"]["task_anchor_order"][self.task]
                count = max(1, int(np.ceil(len(order) * self.curriculum_fraction)))
                anchor_indices = order[:count]
            start = self.sim.random_safe_start(
                self.np_random, self.specification["training_sampler"], goals, self.tolerance,
                target_q=self.tasks[self.task]["q"],
                curriculum_fraction=self.curriculum_fraction, anchor_indices=anchor_indices)
            index = None
            band = None
        else:
            starts = self.starts[self.task] if self.schema == 4 else self.starts
            if index is None:
                index = int(self.np_random.integers(len(starts)))
            if not 0 <= index < len(starts):
                raise ValueError("start_index outside prepared evaluation pool")
            start = starts[index]
            band = self.eval_bands[self.task][index] if self.schema == 4 else None
            self.sim.start_at(start, self.specification["training_sampler"]["clearance"])
        points = self.sim.points()
        if reached(points, self.goal, self.tolerance):
            raise RuntimeError("Initial pose already satisfies task; regenerate nontrivial tasks")
        self.previous_error = goal_errors(points, self.goal)
        self.steps = 0
        self.done = False
        self.frames.clear()
        if self.observation_mode in ("image_state_goal", "image_goal", "image_goal_joints"):
            frame = self.sim.image()
            self.frames.extend([frame.copy() for _ in range(4)])
        return self.observation(), {"task_id": self.task, "start_index": index,
                                    "start_band": band,
                                    "start_q": np.asarray(start, dtype=float).tolist()}

    def step(self, action):
        if self.done:
            raise RuntimeError("reset is required before stepping a terminated/task-switched environment")
        self.sim.advance(action)
        self.steps += 1
        errors = goal_errors(self.sim.points(), self.goal)
        collision = self.sim.collision()
        success = bool(np.all(errors < self.tolerance)) and not collision
        progress = float(10.0 * np.sum(self.previous_error - errors))
        # Preserve original baseline reward components for both methods.
        reward = progress + (100.0 if success else 0.0) - (20.0 if collision else 0.0)
        terminated = collision or success
        truncated = self.steps >= self.max_steps and not terminated
        reason = "collision" if collision else "goal" if success else "timeout" if truncated else None
        self.previous_error = errors
        if self.observation_mode in ("image_state_goal", "image_goal", "image_goal_joints"):
            self.frames.append(self.sim.image())
        self.done = terminated or truncated
        return self.observation(), reward, terminated, truncated, {
            "task_id": self.task, "is_success": success, "collision": collision,
            "cost": float(collision), "done_reason": reason,
            "distance_joint2": float(errors[0]), "distance_ee": float(errors[1]),
        }

    def render(self):
        if not self.frames:
            raise RuntimeError("Rendering requires an image observation mode")
        return np.repeat(self.frames[-1][..., None], 3, axis=2)

    def evaluation_count(self, task):
        if self.split != "eval":
            raise RuntimeError("Only evaluation environments have held-out starts")
        return len(self.starts[task]) if self.schema == 4 else len(self.starts)

    def close(self):
        # Simulator lifetime belongs to the runner; two envs must NOT independently
        # connect to and reset the same scene concurrently.
        pass
