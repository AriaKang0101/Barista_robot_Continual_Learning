"""Build A/B/C from actual forward kinematics and validate on the original scene."""
from __future__ import annotations

import numpy as np

from .core import (SCENE_BLOB, UPSTREAM, UPSTREAM_COMMIT, goal_errors,
                   goal_separation, largest_component, reached, route, validate_tasks, write_json)
from .simulator import Simulator


def drive_route(sim, waypoints, goal, tolerance, max_steps):
    """A validation controller, never a PPO demonstration or training signal."""
    sim.start_at(waypoints[0])
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


def prepare(scene, output, host="localhost", port=23000, seed=2026,
            grid_size=17, tolerance=0.05, clearance=0.01,
            train_starts=6, eval_starts=3, max_steps=500):
    if grid_size < 5 or tolerance <= 0 or clearance < 0 or min(train_starts, eval_starts, max_steps) < 1:
        raise ValueError("Invalid preparation parameters")
    rng = np.random.default_rng(seed)
    with Simulator(scene, host, port) as sim:
        lower, upper = sim.joint_limits()
        axes = [np.linspace(lower[i], upper[i], grid_size) for i in range(2)]
        poses, points = {}, {}
        for i, q1 in enumerate(axes[0]):
            for j, q2 in enumerate(axes[1]):
                q = np.array([q1, q2])
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
        required = train_starts + eval_starts + 3
        if len(component) < required:
            raise RuntimeError(f"Only {len(component)} connected safe poses; need >= {required}. Inspect clearance/grid parameters.")
        # Deterministic farthest-point goal selection within one connected component.
        center = np.mean([np.ravel(points[n]) for n in component], axis=0)
        selected = [max(component, key=lambda n: np.linalg.norm(np.ravel(points[n]) - center))]
        while len(selected) < 3:
            nxt = max(component, key=lambda n: min(goal_separation(points[n], points[g]) for g in selected))
            if min(goal_separation(points[nxt], points[g]) for g in selected) <= 2.2 * tolerance:
                raise RuntimeError("Three distinct success regions do not fit; inspect the robot scale and reduce tolerance appropriately")
            selected.append(nxt)
        candidates = [n for n in component if n not in selected and all(
            np.max(goal_errors(points[n], points[g])) > 2 * tolerance for g in selected)]
        rng.shuffle(candidates)
        accepted, validation = [], []
        # Reject starts whose routes fail under actual dynamics; never silently
        # label a purely kinematic candidate as a usable training task.
        for node in candidates:
            attempts = []
            for label, goal in zip("ABC", selected):
                path = route(graph, node, goal)
                waypoints = np.array([poses[n] for n in path])
                ok, steps, reason = drive_route(sim, waypoints, points[goal], tolerance, max_steps)
                attempts.append({"task": label, "passed": ok, "steps": steps, "reason": reason})
                sim.stop()
                if not ok:
                    break
            print(f"Dynamic checks: start {node}: {attempts}", flush=True)
            validation.append({"node": list(node), "checks": attempts})
            if len(attempts) == 3 and all(v["passed"] for v in attempts):
                accepted.append(node)
            if len(accepted) == train_starts + eval_starts:
                break
        if len(accepted) != train_starts + eval_starts:
            raise RuntimeError("Insufficient dynamically validated starts. No task file was written. Inspect printed failures; do not train invalid tasks.")
        sim.assert_geometry()
        data = {
            "schema_version": 1, "scene_blob": SCENE_BLOB,
            "upstream": UPSTREAM, "upstream_commit": UPSTREAM_COMMIT,
            "runtime": sim.runtime, "fixed_geometry": sim.fixed_geometry,
            "goal_tolerance": tolerance, "max_steps": max_steps, "seed": seed,
            "preparation": {"grid_size": grid_size, "clearance": clearance, "edge_resolution_degrees": 2},
            "tasks": [{"id": label, "q": poses[n].tolist(), "points": points[n]} for label, n in zip("ABC", selected)],
            "train_starts": [poses[n].tolist() for n in accepted[:train_starts]],
            "eval_starts": [poses[n].tolist() for n in accepted[train_starts:]],
            "dynamic_validation": {"all_passed": True, "accepted_nodes": [list(n) for n in accepted], "attempts": validation},
            "preview_paths": {label: [poses[n].tolist() for n in route(graph, accepted[train_starts], goal)]
                              for label, goal in zip("ABC", selected)},
        }
        validate_tasks(data)
        write_json(output, data)
        print(f"Prepared A/B/C and disjoint start pools: {output}", flush=True)

