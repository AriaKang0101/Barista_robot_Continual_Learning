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
    schema = data.get("schema_version")
    if schema not in (2, 3, 4) or data.get("scene_blob") != SCENE_BLOB:
        raise ValueError("Tasks must be prepared from the pinned original scene")
    if schema == 4 and data.get("protocol") != "reachable_state_cl_v4":
        raise ValueError("Invalid protocol v4 identifier")
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
    sampler = data.get("training_sampler", {})
    expected_kind = ("task_conditioned_graph_curriculum" if schema == 4
                     else "continuous_connected_joint_space")
    if sampler.get("kind") != expected_kind:
        raise ValueError("Invalid training sampler kind")
    anchors = np.asarray(sampler.get("anchors", []), dtype=float)
    lower = np.asarray(sampler.get("lower", []), dtype=float)
    upper = np.asarray(sampler.get("upper", []), dtype=float)
    jitter = np.asarray(sampler.get("jitter", []), dtype=float)
    if anchors.ndim != 2 or anchors.shape[1:] != (2,) or len(anchors) < 3 or not np.all(np.isfinite(anchors)):
        raise ValueError("Invalid training sampler anchors")
    if any(v.shape != (2,) or not np.all(np.isfinite(v)) for v in [lower, upper, jitter]):
        raise ValueError("Invalid training sampler bounds")
    if np.any(lower >= upper) or np.any(jitter <= 0) or np.any(anchors < lower) or np.any(anchors > upper):
        raise ValueError("Invalid training sampler geometry")
    if not np.isfinite(sampler.get("clearance", np.nan)) or sampler["clearance"] < 0:
        raise ValueError("Invalid training sampler clearance")
    if not isinstance(sampler.get("max_attempts"), int) or sampler["max_attempts"] < 1:
        raise ValueError("Invalid training sampler attempts")
    exclusion = sampler.get("joint1_exclusion_abs", np.nan)
    if not np.isfinite(exclusion) or exclusion < 0:
        raise ValueError("Invalid joint1 exclusion")
    if schema == 4:
        orders = sampler.get("task_anchor_order", {})
        if set(orders) != {"A", "B", "C"}:
            raise ValueError("Missing per-task curriculum anchor order")
        for task, order in orders.items():
            if (len(order) != len(anchors) or sorted(order) != list(range(len(anchors)))):
                raise ValueError(f"Task {task} curriculum must order every anchor once")
        starts, bands = data.get("eval_starts", {}), data.get("eval_bands", {})
        if set(starts) != {"A", "B", "C"} or set(bands) != {"A", "B", "C"}:
            raise ValueError("Expected task-specific A/B/C evaluation starts and bands")
        for task in "ABC":
            poses = np.asarray(starts[task], dtype=float)
            if (poses.ndim != 2 or poses.shape[1:] != (2,) or len(poses) < 3
                    or not np.all(np.isfinite(poses))):
                raise ValueError(f"Invalid eval_starts for task {task}")
            if len({tuple(q) for q in poses}) != len(poses):
                raise ValueError(f"Task {task} evaluation starts must be distinct")
            if (len(bands[task]) != len(poses)
                    or set(bands[task]) != {"near", "medium", "far"}):
                raise ValueError(f"Invalid evaluation bands for task {task}")
    else:
        poses = np.asarray(data.get("eval_starts", []), dtype=float)
        if poses.ndim != 2 or poses.shape[1:] != (2,) or len(poses) < 3 or not np.all(np.isfinite(poses)):
            raise ValueError("Invalid eval_starts")
        if len({tuple(q) for q in poses}) != len(poses):
            raise ValueError("Evaluation starts must be distinct")
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
    # Legacy configurations used one constant learning rate. Keep them runnable
    # while v4 uses a per-task linear schedule.
    if "learning_rate_start" not in cfg and "learning_rate" in cfg:
        cfg["learning_rate_start"] = cfg["learning_rate"]
    if "learning_rate_end" not in cfg and "learning_rate" in cfg:
        cfg["learning_rate_end"] = cfg["learning_rate"]
    for name in ["timesteps_per_task", "n_steps", "batch_size", "n_epochs", "fisher_samples"]:
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
    for name in ["learning_rate_start", "learning_rate_end"]:
        if not isinstance(cfg.get(name), (int, float)) or not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg["learning_rate_end"] > cfg["learning_rate_start"]:
        raise ValueError("learning_rate_end cannot exceed learning_rate_start")
    mode = cfg.setdefault("observation_mode", "state_goal")
    if mode not in ("state_goal", "image_state_goal", "image_goal", "image_goal_joints"):
        raise ValueError("Invalid observation_mode")
    initial = cfg.get("curriculum_initial_fraction")
    full = cfg.get("curriculum_full_fraction")
    if not isinstance(initial, (int, float)) or not 0 < initial <= 1:
        raise ValueError("curriculum_initial_fraction must be in (0, 1]")
    if not isinstance(full, (int, float)) or not 0 < full <= 1:
        raise ValueError("curriculum_full_fraction must be in (0, 1]")
    if sorted(cfg["task_order"]) != ["A", "B", "C"]:
        raise ValueError("task_order must contain A, B, C once each")
    return cfg
