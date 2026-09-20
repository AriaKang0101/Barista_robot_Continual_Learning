"""Prepare reachable, difficulty-controlled A/B/C tasks for protocol v4."""
from __future__ import annotations

from collections import deque

import numpy as np

from .core import (SCENE_BLOB, UPSTREAM, UPSTREAM_COMMIT, goal_separation,
                   largest_component, reached, route, validate_tasks, write_json)
from .simulator import Simulator


def graph_distances(graph, source):
    distances = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nxt in sorted(graph[node]):
            if nxt not in distances:
                distances[nxt] = distances[node] + 1
                queue.append(nxt)
    return distances


def drive_route(sim, waypoints, goal, tolerance, max_steps, clearance=0.0):
    """Validation controller only; never PPO training data or demonstrations."""
    sim.start_at(waypoints[0], clearance)
    waypoint = 1 if len(waypoints) > 1 else 0
    for step in range(max_steps):
        if sim.collision():
            return False, step, "collision"
        if reached(sim.points(), goal, tolerance):
            return True, step, "goal"
        q = sim.joint_positions()
        while waypoint < len(waypoints) - 1 and np.max(np.abs(waypoints[waypoint] - q)) < 0.045:
            waypoint += 1
        sim.advance(3.0 * (waypoints[waypoint] - q) / 0.8)
    if not sim.collision() and reached(sim.points(), goal, tolerance):
        return True, max_steps, "goal"
    return False, max_steps, "timeout"


def select_goals(sim, graph, poses, points, component, lower, upper,
                 tolerance, goal_clearance):
    """Choose three distinct, central, well-connected and reachable goals."""
    robust = []
    for node in component:
        if len(graph[node]) >= 3 and sim.safe_pose(poses[node], goal_clearance):
            robust.append(node)
    if len(robust) < 6:
        raise RuntimeError("Too few high-clearance interior goal candidates")

    scale = np.asarray(upper) - np.asarray(lower)
    normalized = {n: (np.asarray(poses[n]) - lower) / scale for n in robust}
    center = np.mean([normalized[n] for n in robust], axis=0)
    radius = {n: float(np.linalg.norm(normalized[n] - center)) for n in robust}
    cutoff = float(np.quantile(list(radius.values()), 0.75))
    central = [n for n in robust if radius[n] <= cutoff]
    selected = [min(central, key=lambda n: (radius[n], n))]
    while len(selected) < 3:
        eligible = [n for n in central if n not in selected
                    and all(goal_separation(points[n], points[g]) > 2.2 * tolerance
                            for g in selected)]
        if not eligible:
            eligible = [n for n in robust if n not in selected
                        and all(goal_separation(points[n], points[g]) > 2.2 * tolerance
                                for g in selected)]
        if not eligible:
            raise RuntimeError("Three separated, high-clearance reachable goals do not fit")
        selected.append(max(
            eligible,
            key=lambda n: (min(np.linalg.norm(normalized[n] - normalized[g]) for g in selected),
                           -radius[n], n)))
    return selected


def prepare(scene, output, host="localhost", port=23000, seed=2026,
            grid_size=17, tolerance=0.15, clearance=0.01,
            goal_clearance=0.03, eval_starts=30, max_steps=500):
    if (grid_size < 5 or tolerance <= 0 or clearance < 0 or goal_clearance < clearance
            or eval_starts < 6 or max_steps < 1):
        raise ValueError("Invalid preparation parameters")
    rng = np.random.default_rng(seed)
    with Simulator(scene, host, port) as sim:
        lower, upper = sim.joint_limits()
        lower_array, upper_array = np.asarray(lower), np.asarray(upper)
        axes = [np.linspace(lower[i], upper[i], grid_size) for i in range(2)]
        jitter = np.asarray([(axis[1] - axis[0]) / 2 for axis in axes])
        poses, points = {}, {}
        for i, q1 in enumerate(axes[0]):
            for j, q2 in enumerate(axes[1]):
                q = np.asarray([q1, q2])
                if sim.safe_pose(q, clearance):
                    poses[i, j] = q
                    points[i, j] = sim.points().tolist()
            print(f"Geometry sampling: {i + 1}/{grid_size} rows, {len(poses)} safe poses", flush=True)

        graph = {node: set() for node in poses}
        for index, (i, j) in enumerate(sorted(poses)):
            for nxt in [(i + 1, j), (i, j + 1)]:
                if nxt not in poses:
                    continue
                a, b = poses[i, j], poses[nxt]
                count = max(2, int(np.ceil(np.max(np.abs(b - a)) / np.deg2rad(2))) + 1)
                if all(sim.safe_pose(q, clearance) for q in np.linspace(a, b, count)[1:-1]):
                    graph[i, j].add(nxt)
                    graph[nxt].add((i, j))
            if index % 20 == 0:
                print(f"Path checks: {index + 1}/{len(poses)} nodes", flush=True)

        component = largest_component(graph)
        if len(component) < 12:
            raise RuntimeError("The connected safe component is too small")
        goal_nodes = select_goals(sim, graph, poses, points, component, lower_array,
                                  upper_array, tolerance, goal_clearance)
        labels = list("ABC")
        tasks = [{"id": label, "q": poses[node].tolist(), "points": points[node],
                  "graph_node": list(node)}
                 for label, node in zip(labels, goal_nodes)]
        print("Selected reachable goals: " + ", ".join(
            f"{task['id']} q={np.round(task['q'], 4).tolist()}" for task in tasks), flush=True)

        anchor_nodes = list(component)
        anchors = np.asarray([poses[n] for n in anchor_nodes])
        anchor_index = {node: i for i, node in enumerate(anchor_nodes)}
        distances = {label: graph_distances(graph, node)
                     for label, node in zip(labels, goal_nodes)}
        task_anchor_order = {
            label: [anchor_index[n] for n in sorted(
                anchor_nodes, key=lambda n: (distances[label][n], n))]
            for label in labels
        }
        sampler = {
            "kind": "task_conditioned_graph_curriculum",
            "anchors": anchors.tolist(),
            "lower": lower_array.tolist(), "upper": upper_array.tolist(),
            "jitter": jitter.tolist(), "clearance": clearance,
            "joint1_exclusion_abs": 0.0, "max_attempts": 200,
            "task_anchor_order": task_anchor_order,
        }

        eval_by_task, bands_by_task, validation, preview_paths = {}, {}, [], {}
        band_names = ["near", "medium", "far"]
        base_count, remainder = divmod(eval_starts, 3)
        required = {name: base_count + int(i < remainder) for i, name in enumerate(band_names)}

        for label, goal_node, task in zip(labels, goal_nodes, tasks):
            goal = np.asarray(task["points"])
            eligible = [n for n in sorted(anchor_nodes, key=lambda n: (distances[label][n], n))
                        if n != goal_node and not reached(points[n], goal, tolerance)]
            split_indexes = np.array_split(np.arange(len(eligible)), 3)
            pools = {name: [eligible[int(i)] for i in split]
                     for name, split in zip(band_names, split_indexes)}
            accepted, accepted_bands, accepted_nodes = [], [], []
            for band in band_names:
                attempts = 0
                limit = max(1000, required[band] * 200)
                while accepted_bands.count(band) < required[band] and attempts < limit:
                    attempts += 1
                    pool = pools[band]
                    if not pool:
                        break
                    node = pool[int(rng.integers(len(pool)))]
                    anchor = poses[node]
                    q = np.clip(anchor + rng.uniform(-jitter, jitter), lower_array, upper_array)
                    sim.stop()
                    if not sim.safe_pose(q, clearance):
                        continue
                    if reached(sim.points(), goal, tolerance):
                        continue
                    count = max(2, int(np.ceil(np.max(np.abs(anchor - q)) / np.deg2rad(2))) + 1)
                    if not all(sim.safe_pose(p, clearance) for p in np.linspace(q, anchor, count)):
                        continue
                    if any(np.max(np.abs(q - prior)) < 1e-6 for prior in accepted):
                        continue
                    path = route(graph, node, goal_node)
                    waypoints = np.vstack([q, [poses[n] for n in path]])
                    ok, steps, reason = drive_route(
                        sim, waypoints, goal, tolerance, max_steps, clearance)
                    sim.stop()
                    validation.append({"task": label, "band": band, "q": q.tolist(),
                                       "anchor_node": list(node), "passed": ok,
                                       "steps": steps, "reason": reason})
                    if not ok:
                        continue
                    accepted.append(q.copy())
                    accepted_bands.append(band)
                    accepted_nodes.append(node)
                    print(f"Dynamic validation: Task {label} {band} "
                          f"{accepted_bands.count(band)}/{required[band]}", flush=True)
            if len(accepted) != eval_starts:
                raise RuntimeError(f"Insufficient validated evaluation starts for Task {label}")
            eval_by_task[label] = [q.tolist() for q in accepted]
            bands_by_task[label] = accepted_bands
            first_node = accepted_nodes[0]
            preview_paths[label] = np.vstack([
                accepted[0], [poses[n] for n in route(graph, first_node, goal_node)]]).tolist()

        sim.assert_geometry()
        data = {
            "schema_version": 4, "protocol": "reachable_state_cl_v4",
            "scene_blob": SCENE_BLOB, "upstream": UPSTREAM,
            "upstream_commit": UPSTREAM_COMMIT,
            "runtime": sim.runtime, "fixed_geometry": sim.fixed_geometry,
            "goal_tolerance": tolerance, "max_steps": max_steps, "seed": seed,
            "preparation": {"grid_size": grid_size, "clearance": clearance,
                            "goal_clearance": goal_clearance,
                            "edge_resolution_degrees": 2,
                            "evaluation_bands": required},
            "tasks": tasks, "training_sampler": sampler,
            "eval_starts": eval_by_task, "eval_bands": bands_by_task,
            "dynamic_validation": {"all_passed": True, "evaluation_attempts": validation},
            "preview_paths": preview_paths,
        }
        validate_tasks(data)
        write_json(output, data)
        print(f"Prepared protocol v4: 3 reachable goals and {eval_starts} starts per task: {output}",
              flush=True)
