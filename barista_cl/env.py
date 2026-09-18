"""Identical goal-conditioned environment for both learning methods."""
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import goal_errors, reached, validate_tasks


class BaristaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, simulator, tasks, split="train"):
        super().__init__()
        validate_tasks(tasks)
        if split not in ("train", "eval"):
            raise ValueError("split must be train or eval")
        self.sim = simulator
        self.specification = tasks
        self.tasks = {t["id"]: t for t in tasks["tasks"]}
        self.starts = np.asarray(tasks["eval_starts"], dtype=float) if split == "eval" else None
        self.split = split
        self.tolerance = tasks["goal_tolerance"]
        self.max_steps = tasks["max_steps"]
        sampler = tasks["training_sampler"]
        self.joint_lower = np.asarray(sampler["lower"], dtype=np.float32)
        self.joint_upper = np.asarray(sampler["upper"], dtype=np.float32)
        self.curriculum_fraction = 1.0
        self.task = "A"
        self.frames = deque(maxlen=4)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        # Stack only images, NOT goal coordinates. Image order is explicitly CHW.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (4, 84, 84), np.uint8),
            "goal": spaces.Box(-np.inf, np.inf, (6,), np.float32),
            "joints": spaces.Box(-1.0, 1.0, (2,), np.float32),
        })
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
        return {"image": np.stack(self.frames),
                "goal": (self.goal - self.base_position).astype(np.float32).ravel(),
                "joints": np.clip(normalized_q, -1.0, 1.0).astype(np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        index = options.get("start_index")
        if self.split == "train":
            if index is not None:
                raise ValueError("Training uses continuous safe random starts, not start_index")
            goals = [t["points"] for t in self.tasks.values()]
            start = self.sim.random_safe_start(
                self.np_random, self.specification["training_sampler"], goals, self.tolerance,
                target_q=self.tasks[self.task]["q"],
                curriculum_fraction=self.curriculum_fraction)
            index = None
        else:
            if index is None:
                index = int(self.np_random.integers(len(self.starts)))
            if not 0 <= index < len(self.starts):
                raise ValueError("start_index outside prepared evaluation pool")
            start = self.starts[index]
            self.sim.start_at(start, self.specification["training_sampler"]["clearance"])
        points = self.sim.points()
        if reached(points, self.goal, self.tolerance):
            raise RuntimeError("Initial pose already satisfies task; regenerate nontrivial tasks")
        self.previous_error = goal_errors(points, self.goal)
        self.steps = 0
        self.done = False
        frame = self.sim.image()
        self.frames.clear()
        self.frames.extend([frame.copy() for _ in range(4)])
        return self.observation(), {"task_id": self.task, "start_index": index,
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
        self.frames.append(self.sim.image())
        self.done = terminated or truncated
        return self.observation(), reward, terminated, truncated, {
            "task_id": self.task, "is_success": success, "collision": collision,
            "cost": float(collision), "done_reason": reason,
            "distance_joint2": float(errors[0]), "distance_ee": float(errors[1]),
        }

    def render(self):
        return np.repeat(self.frames[-1][..., None], 3, axis=2)

    def close(self):
        # Simulator lifetime belongs to the runner; two envs must NOT independently
        # connect to and reset the same scene concurrently.
        pass
