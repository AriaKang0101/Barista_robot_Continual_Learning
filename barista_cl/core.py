"""Serializable experimental protocol and pure geometry/metric helpers."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np

UPSTREAM = "ymk2001/Barista_robot_Reinforcement_Learning"
UPSTREAM_COMMIT = "232dd870ff50f988b1168d7630998438964f5423"
SCENE_BLOB = "925ea5fcc4c2ecb8cb93c1d8cc11e68c434c7b53"
SCENE_NAME = "safety_rl_2dof.ttt"


def git_blob_sha(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def goal_errors(points, goal):
    return np.linalg.norm(np.asarray(points) - np.asarray(goal), axis=1)


def reached(points, goal, tolerance):
    return bool(np.all(goal_errors(points, goal) < tolerance))


def goal_separation(a, b):
    # AND success regions are disjoint if at least one of the two point pairs
    # is separated by more than twice the position tolerance.
    return float(np.max(goal_errors(a, b)))


def largest_component(graph):
    remaining = set(graph)
    best = set()
    while remaining:
        root = min(remaining)
        found = {root}
        queue = deque([root])
        while queue:
            for nxt in graph[queue.popleft()]:
                if nxt not in found:
                    found.add(nxt)
                    queue.append(nxt)
        remaining -= found
        if len(found) > len(best):
            best = found
    return sorted(best)


def route(graph, source, target):
    parents = {source: None}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return path[::-1]
        for nxt in sorted(graph[node]):
            if nxt not in parents:
                parents[nxt] = node
                queue.append(nxt)
    raise ValueError("No collision-free sampled joint-space path exists")


def validate_tasks(data):
    if data.get("schema_version") != 1 or data.get("scene_blob") != SCENE_BLOB:
        raise ValueError("Tasks must be prepared from the pinned original scene")
    tasks = data.get("tasks", [])
    if len(tasks) != 3 or {t["id"] for t in tasks} != {"A", "B", "C"}:
        raise ValueError("Expected three task IDs: A, B, C")
    tol = float(data["goal_tolerance"])
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("goal_tolerance must be finite and positive")
    for t in tasks:
        p, q = np.asarray(t["points"]), np.asarray(t["q"])
        if p.shape != (2, 3) or q.shape != (2,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise ValueError("Invalid target geometry")
    for i in range(3):
        for j in range(i):
            if goal_separation(tasks[i]["points"], tasks[j]["points"]) <= 2 * tol:
                raise ValueError("Overlapping task success regions")
    for split in ["train_starts", "eval_starts"]:
        poses = np.asarray(data[split])
        if poses.ndim != 2 or poses.shape[1] != 2 or len(poses) == 0 or not np.all(np.isfinite(poses)):
            raise ValueError(f"Invalid {split}")
    if {tuple(q) for q in data["train_starts"]} & {tuple(q) for q in data["eval_starts"]}:
        raise ValueError("Training and evaluation starts overlap")
    if not data.get("dynamic_validation", {}).get("all_passed"):
        raise ValueError("Run prepare: every start/goal pair must pass dynamic validation")


def continual_metrics(matrix):
    """Rows: completed stages; columns: tasks in training order. NaN is untested."""
    r = np.asarray(matrix, dtype=float)
    n = len(r)
    if r.shape != (n, n) or n == 0:
        raise ValueError("Expected square performance matrix")
    final = r[-1]
    if not np.all(np.isfinite(final)):
        raise ValueError("Final stage must evaluate every task")
    forgetting, bwt = [], []
    for j in range(n - 1):
        past = r[j:n - 1, j]
        if not np.all(np.isfinite(past)):
            raise ValueError("Missing learned-task evaluation")
        forgetting.append(float(np.max(past) - final[j]))
        bwt.append(float(final[j] - r[j, j]))
    return {
        "final_mean_success": float(np.mean(final)),
        "mean_forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "backward_transfer": float(np.mean(bwt)) if bwt else 0.0,
    }


def load_config(path):
    cfg = read_json(path)
    for name in ["timesteps_per_task", "n_steps", "batch_size", "n_epochs", "fisher_samples", "evaluation_episodes"]:
        if not isinstance(cfg[name], int) or cfg[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["batch_size"] < 2 or cfg["n_steps"] % cfg["batch_size"]:
        raise ValueError("batch_size >= 2 must divide n_steps")
    if cfg["timesteps_per_task"] % cfg["n_steps"]:
        raise ValueError("timesteps_per_task must be a multiple of n_steps for an exact shared budget")
    if cfg["fisher_samples"] > cfg["n_steps"]:
        raise ValueError("fisher_samples cannot exceed the final rollout size")
    if cfg["ewc_lambda"] < 0 or not np.isfinite(cfg["ewc_lambda"]):
        raise ValueError("ewc_lambda must be finite and nonnegative")
    if sorted(cfg["task_order"]) != ["A", "B", "C"]:
        raise ValueError("task_order must contain A, B, C once each")
    return cfg
