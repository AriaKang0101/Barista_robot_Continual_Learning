"""Build A/B/C from actual forward kinematics and validate on the original scene."""
from __future__ import annotations

import numpy as np

from .core import (SCENE_BLOB, UPSTREAM, UPSTREAM_COMMIT, goal_errors,
                   goal_separation, largest_component, reached, route, validate_tasks, write_json)
from .simulator import Simulator


def drive_route(sim, waypoints, goal, tolerance, max_steps, clearance=0.0):
    """A validation controller, never a PPO demonstration or training signal."""
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


def prepare(scene, output, host="localhost", port=23000, seed=2026,
            grid_size=17, tolerance=0.05, clearance=0.01,
            eval_starts=30, max_steps=500):
    if grid_size < 5 or tolerance <= 0 or clearance < 0 or min(eval_starts, max_steps) < 1:
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
        if len(component) < 6:
            raise RuntimeError("The connected safe component is too small; inspect clearance/grid parameters.")
        # Deterministic farthest-point goal selection within one connected component.
        center = np.mean([np.ravel(points[n]) for n in component], axis=0)
        selected = [max(component, key=lambda n: np.linalg.norm(np.ravel(points[n]) - center))]
        while len(selected) < 3:
            nxt = max(component, key=lambda n: min(goal_separation(points[n], points[g]) for g in selected))
            if min(goal_separation(points[nxt], points[g]) for g in selected) <= 2.2 * tolerance:
                raise RuntimeError("Three distinct success regions do not fit; inspect the robot scale and reduce tolerance appropriately")
            selected.append(nxt)
        lower_array, upper_array = np.asarray(lower), np.asarray(upper)
        jitter = np.asarray([(axis[1] - axis[0]) / 2 for axis in axes])
        anchors = np.asarray([poses[n] for n in component])
        sampler = {
            "kind": "continuous_connected_joint_space",
            "anchors": anchors.tolist(),
            "lower": lower_array.tolist(), "upper": upper_array.tolist(),
            "jitter": jitter.tolist(), "clearance": clearance,
            "joint1_exclusion_abs": float(np.deg2rad(10)), "max_attempts": 100,
        }
        accepted, accepted_anchors, validation = [], [], []
        # Evaluation starts are random and diverse, but generated once and
        # reused by every method/seed. Each candidate is connected locally to
        # the safe component and dynamically validated against every goal.
        candidate_attempts = 0
        max_attempts = max(1000, eval_starts * 100)
        while len(accepted) < eval_starts and candidate_attempts < max_attempts:
            candidate_attempts += 1
            node = component[int(rng.integers(len(component)))]
            anchor = poses[node]
            q = np.clip(anchor + rng.uniform(-jitter, jitter), lower_array, upper_array)
            if abs(q[0]) < sampler["joint1_exclusion_abs"]:
                continue
            sim.stop()
            if not sim.safe_pose(q, clearance):
                continue
            candidate_points = sim.points()
            if any(reached(candidate_points, points[g], tolerance) for g in selected):
                continue
            count = max(2, int(np.ceil(np.max(np.abs(anchor - q)) / np.deg2rad(2))) + 1)
            if not all(sim.safe_pose(p, clearance) for p in np.linspace(q, anchor, count)):
                continue
            if any(np.max(np.abs(q - prior)) < 1e-6 for prior in accepted):
                continue
            checks = []
            for label, goal in zip("ABC", selected):
                path = route(graph, node, goal)
                waypoints = np.vstack([q, [poses[n] for n in path]])
                ok, steps, reason = drive_route(sim, waypoints, points[goal], tolerance, max_steps, clearance)
                checks.append({"task": label, "passed": ok, "steps": steps, "reason": reason})
                sim.stop()
                if not ok:
                    break
            print(f"Dynamic checks: eval start {len(accepted) + 1}/{eval_starts}: {checks}", flush=True)
            validation.append({"q": q.tolist(), "anchor_node": list(node), "checks": checks})
            if len(checks) == 3 and all(v["passed"] for v in checks):
                accepted.append(q)
                accepted_anchors.append(node)
        if len(accepted) != eval_starts:
            raise RuntimeError("Insufficient dynamically validated evaluation starts. No task file was written.")
        sim.assert_geometry()
        data = {
            "schema_version": 2, "scene_blob": SCENE_BLOB,
            "upstream": UPSTREAM, "upstream_commit": UPSTREAM_COMMIT,
            "runtime": sim.runtime, "fixed_geometry": sim.fixed_geometry,
            "goal_tolerance": tolerance, "max_steps": max_steps, "seed": seed,
            "preparation": {"grid_size": grid_size, "clearance": clearance, "edge_resolution_degrees": 2},
            "tasks": [{"id": label, "q": poses[n].tolist(), "points": points[n]} for label, n in zip("ABC", selected)],
            "training_sampler": sampler,
            "eval_starts": [q.tolist() for q in accepted],
            "dynamic_validation": {"all_passed": True, "evaluation_attempts": validation},
            "preview_paths": {label: np.vstack([accepted[0], [poses[n] for n in route(
                                  graph, accepted_anchors[0], goal)]]).tolist()
                              for label, goal in zip("ABC", selected)},
        }
        validate_tasks(data)
        write_json(output, data)
        print(f"Prepared A/B/C, continuous safe training sampler, and {eval_starts} held-out starts: {output}", flush=True)
