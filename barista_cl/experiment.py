"""Stage-boundary training/evaluation with matched tasks, seeds and budgets."""
from __future__ import annotations

import csv
import importlib.metadata
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .core import continual_metrics, digest, load_config, read_json, validate_tasks, write_json
from .env import BaristaEnv
from .ewc import EWCPPO, preserve_rng
from .simulator import Simulator


class EpisodeLog(BaseCallback):
    def __init__(self, path, stage, task, env, stage_start, budget,
                 initial_fraction, full_fraction, learning_rate_schedule):
        super().__init__()
        self.path, self.stage, self.task = Path(path), stage, task
        self.env, self.stage_start, self.budget = env, stage_start, budget
        self.initial_fraction, self.full_fraction = initial_fraction, full_fraction
        self.learning_rate_schedule = learning_rate_schedule

    def _on_step(self):
        progress = min(1.0, max(0.0, (self.model.num_timesteps - self.stage_start) / self.budget))
        expansion = min(1.0, progress / self.full_fraction)
        self.learning_rate_schedule.progress_remaining = 1.0 - progress
        self.env.set_curriculum_fraction(
            self.initial_fraction + (1.0 - self.initial_fraction) * expansion)
        info = self.locals["infos"][0]
        if self.locals["dones"][0]:
            append_csv(self.path, {
                "stage": self.stage, "task": self.task, "train_steps": self.num_timesteps,
                "return": info["episode"]["r"], "length": info["episode"]["l"],
                "success": int(info["is_success"]), "collision": int(info["collision"]),
                "reason": info["done_reason"],
            })
        return True


class PerTaskLinearSchedule:
    """Linear LR decay that restarts at each task boundary.

    SB3's normal progress value spans repeated ``learn`` calls in a way that
    would make the rate jump between continual-learning stages. The callback
    explicitly supplies progress within the current, equal-budget task.
    """
    def __init__(self, start, end):
        self.start = float(start)
        self.end = float(end)
        self.progress_remaining = 1.0

    def reset(self):
        self.progress_remaining = 1.0

    def __call__(self, _):
        return self.end + (self.start - self.end) * self.progress_remaining


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate(model, env, task_ids, output, stage):
    """Frozen deterministic evaluation on held-out initial configurations."""
    rows = []
    previous_mode = model.policy.training
    with preserve_rng():
        model.policy.set_training_mode(False)
        try:
            for task in task_ids:
                env.set_task(task)
                # Every held-out random start is used exactly once. The saved
                # bank is shared by all methods and training seeds.
                for episode, start in enumerate(range(len(env.starts))):
                    obs, _ = env.reset(seed=100000 + episode, options={"start_index": start})
                    total = 0.0
                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        total += reward
                        if terminated or truncated:
                            break
                    row = {
                        "stage": stage, "task": task, "episode": episode,
                        "start_index": start, "success": int(info["is_success"]),
                        "collision": int(info["collision"]), "timeout": int(info["done_reason"] == "timeout"),
                        "steps": env.steps, "return": total,
                        "distance_joint2": info["distance_joint2"], "distance_ee": info["distance_ee"],
                    }
                    append_csv(output, row)
                    rows.append(row)
        finally:
            model.policy.set_training_mode(previous_mode)
    return rows


def versions():
    names = ["torch", "stable-baselines3", "gymnasium", "numpy", "opencv-python-headless", "coppeliasim-zmqremoteapi-client"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in names},
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def run(scene, tasks_path, config_path, output, method, seed=None, device=None,
        host="localhost", port=23000, only_task=None):
    config, tasks = load_config(config_path), read_json(tasks_path)
    validate_tasks(tasks)
    if method not in ("sequential", "ewc"):
        raise ValueError("method must be sequential or ewc")
    if seed is not None:
        config["seed"] = seed
    if device is not None:
        config["device"] = device
    order = [only_task] if only_task else config["task_order"]
    if any(t not in "ABC" for t in order):
        raise ValueError("Invalid task")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix/overwrite experiments in {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Deterministic torch kernels where supported; physical simulation itself
    # can still vary between operating systems/physics engines.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    write_json(output / "tasks.json", tasks)
    write_json(output / "config.json", config)
    manifest = {"method": method, "seed": seed, "task_order": order,
                "tasks_sha256": digest(tasks), "versions": versions(),
                "status": "running", "actual_training_steps": 0,
                "fisher_source": "final current-task rollout; no extra environment interaction"}
    write_json(output / "manifest.json", manifest)
    matrix = np.full((len(order), len(order)), np.nan)
    start_time = time.monotonic()
    try:
        with Simulator(scene, host, port) as sim:
            train_env = BaristaEnv(sim, tasks, "train", config["observation_mode"])
            eval_env = BaristaEnv(sim, tasks, "eval", config["observation_mode"])
            vector = DummyVecEnv([lambda: Monitor(train_env)])
            learning_rate = PerTaskLinearSchedule(config["learning_rate_start"],
                                                  config["learning_rate_end"])
            args = {k: config[k] for k in ["n_steps", "batch_size", "n_epochs", "gamma",
                                            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm"]}
            model = EWCPPO("MultiInputPolicy", vector,
                           ewc_lambda=config["ewc_lambda"] if method == "ewc" else 0.0,
                           policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                           seed=seed, device=config["device"], verbose=1,
                           tensorboard_log=str(output / "tensorboard"),
                           learning_rate=learning_rate, **args)
            manifest["resolved_device"] = str(model.device)
            for stage, task in enumerate(order, start=1):
                train_env.set_task(task)
                train_env.set_curriculum_fraction(config["curriculum_initial_fraction"])
                learning_rate.reset()
                # Evaluation shares the simulator, so never reuse an old rollout
                # observation. set_env(force_reset=True) resets frame/episode state.
                model.set_env(vector, force_reset=True)
                vector.seed(seed + 1009 * stage)
                begin = model.num_timesteps
                callback = EpisodeLog(
                    output / "training_episodes.csv", stage, task, train_env, begin,
                    config["timesteps_per_task"], config["curriculum_initial_fraction"],
                    config["curriculum_full_fraction"], learning_rate)
                model.learn(total_timesteps=config["timesteps_per_task"], reset_num_timesteps=False,
                            tb_log_name=method, callback=callback)
                assert model.num_timesteps - begin == config["timesteps_per_task"]
                if method == "ewc":
                    model.consolidate(task, config["fisher_samples"], seed + 7001 * stage)
                checkpoint = output / f"stage_{stage}_{task}.zip"
                # EWC anchors and Fisher tensors are included in SB3 model.save().
                model.save(checkpoint)
                rows = evaluate(model, eval_env, order[:stage],
                                output / "evaluation_episodes.csv", stage)
                for j, old_task in enumerate(order[:stage]):
                    matrix[stage - 1, j] = np.mean([r["success"] for r in rows if r["task"] == old_task])
                np.savetxt(output / "success_matrix.csv", matrix, delimiter=",", header=",".join(order), comments="")
                manifest["actual_training_steps"] = model.num_timesteps
                manifest["completed_stages"] = stage
                write_json(output / "manifest.json", manifest)
            vector.close()
        write_json(output / "metrics.json", continual_metrics(matrix))
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - start_time
        write_json(output / "manifest.json", manifest)
    print(f"Completed: {output}", flush=True)
