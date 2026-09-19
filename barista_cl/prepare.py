"""Build A/B/C from actual forward kinematics and validate on the original scene."""
from __future__ import annotations

import numpy as np

from .core import (SCENE_BLOB, UPSTREAM, UPSTREAM_COMMIT, goal_errors,
                   goal_separation, largest_component, reached, route, validate_tasks, write_json)
from .simulator import Simulator


def refine_original_goal(sim, graph, poses, points, component, goal, tolerance, jitter,
                         lower, upper, subdivisions=21, candidate_anchors=8,
                         clearance=0.0):
    """Find a continuous safe q for the original markers between coarse grid nodes.

    A 17x17 grid is sufficient for connectivity, but a long second link can move
    more than the 0.15 m success radius between adjacent samples. Refine around
    the most promising connected anchors instead of incorrectly declaring the
    original task unreachable from the coarse samples alone.
    """
    ranked = sorted(component,
                    key=lambda n: float(np.max(goal_errors(points[n], goal))))
    offsets = [np.asarray([a, b]) * jitter
               for a in np.linspace(-1.0, 1.0, subdivisions)
               for b in np.linspace(-1.0, 1.0, subdivisions)]
    offsets.sort(key=lambda delta: (float(np.dot(delta, delta)), float(delta[0]), float(delta[1])))
    best = None
    for node in ranked[:min(candidate_anchors, len(ranked))]:
        anchor = np.asarray(poses[node])
        for offset in offsets:
            q = np.clip(anchor + offset, lower, upper)
            if not sim.safe_pose(q, clearance):
                continue
            candidate_points = sim.points()
            errors = goal_errors(candidate_points, goal)
            score = float(np.max(errors))
            if best is not None and score >= best[0]:
                continue
            count = max(2, int(np.ceil(np.max(np.abs(anchor - q)) / np.deg2rad(2))) + 1)
            if not all(sim.safe_pose(p, clearance) for p in np.linspace(anchor, q, count)):
                continue
            best = (score, q.copy(), node, errors.copy())
    if best is None or not np.all(best[3] < tolerance):
        errors = None if best is None else best[3]
        raise RuntimeError(
            "Original target markers are not reachable after continuous safe refinement: "
            f"best_errors={errors}. Do not change tolerance; inspect the scene/target geometry."
        )
    return best[1], best[2], best[3]


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
            grid_size=17, tolerance=0.15, clearance=0.01,
            eval_starts=30, max_steps=500):
    if grid_size < 5 or tolerance <= 0 or clearance < 0 or min(eval_starts, max_steps) < 1:
        raise ValueError("Invalid preparation parameters")
    rng = np.random.default_rng(seed)
    with Simulator(scene, host, port) as sim:
        original_goal = sim.original_target_points()
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
        # Task A is the exact target pair used by the pinned upstream PPO scene.
        # Its q is only a collision-free route endpoint/reference; success is
        # always measured against the untouched FirstTarget/EndTarget markers.
        center = np.mean([np.ravel(points[n]) for n in component], axis=0)
        lower_array, upper_array = np.asarray(lower), np.asarray(upper)
        jitter = np.asarray([(axis[1] - axis[0]) / 2 for axis in axes])
        task_a_q, task_a, task_a_errors = refine_original_goal(
            sim, graph, poses, points, component, original_goal, tolerance,
            jitter, lower_array, upper_array, clearance=clearance)
        print(f"Original Task A continuous refinement: q={task_a_q.tolist()}, "
              f"errors={task_a_errors.tolist()}", flush=True)

        # B/C are central, well-connected safe goals rather than the extreme
        # farthest-point goals used by v2. This avoids confounding continual
        # learning with three unnecessarily hard boundary-reaching problems.
        span = np.asarray(upper) - np.asarray(lower)
        margin = 0.5 / (grid_size - 1)
        interior = [n for n in component
                    if len(graph[n]) >= 3
                    and np.min(np.minimum((poses[n] - lower) / span,
                                          (upper - poses[n]) / span)) >= margin]
        candidates = sorted(interior or component,
                            key=lambda n: (np.linalg.norm(np.ravel(points[n]) - center), n))
        selected = [task_a]
        task_points = [original_goal.tolist()]
        for node in candidates:
            if node in selected:
                continue
            if all(goal_separation(points[node], goal) > 2.2 * tolerance for goal in task_points):
                selected.append(node)
                task_points.append(points[node])
            if len(selected) == 3:
                break
        if len(selected) != 3:
            raise RuntimeError("Three distinct interior success regions do not fit; inspect grid/tolerance parameters")
        task_qs = [task_a_q] + [np.asarray(poses[n]) for n in selected[1:]]
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
            if any(reached(candidate_points, goal, tolerance) for goal in task_points):
                continue
            count = max(2, int(np.ceil(np.max(np.abs(anchor - q)) / np.deg2rad(2))) + 1)
            if not all(sim.safe_pose(p, clearance) for p in np.linspace(q, anchor, count)):
                continue
            if any(np.max(np.abs(q - prior)) < 1e-6 for prior in accepted):
                continue
            checks = []
            for label, goal_node, goal_q, goal_points in zip("ABC", selected, task_qs, task_points):
                path = route(graph, node, goal_node)
                waypoints = np.vstack([q, [poses[n] for n in path], goal_q])
                ok, steps, reason = drive_route(sim, waypoints, goal_points, tolerance, max_steps, clearance)
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
            "schema_version": 3, "scene_blob": SCENE_BLOB,
            "upstream": UPSTREAM, "upstream_commit": UPSTREAM_COMMIT,
            "runtime": sim.runtime, "fixed_geometry": sim.fixed_geometry,
            "goal_tolerance": tolerance, "max_steps": max_steps, "seed": seed,
            "preparation": {"grid_size": grid_size, "clearance": clearance, "edge_resolution_degrees": 2},
            "goal_selection": {
                "A": "original_scene_markers",
                "B_C": "central_well_connected_safe_goals",
            },
            "tasks": [{"id": label, "q": np.asarray(goal_q).tolist(),
                       "points": np.asarray(goal).tolist()}
                      for label, goal_q, goal in zip("ABC", task_qs, task_points)],
            "training_sampler": sampler,
            "eval_starts": [q.tolist() for q in accepted],
            "dynamic_validation": {"all_passed": True, "evaluation_attempts": validation},
            "preview_paths": {label: np.vstack([accepted[0], [poses[n] for n in route(
                                  graph, accepted_anchors[0], goal)], goal_q]).tolist()
                              for label, goal, goal_q in zip("ABC", selected, task_qs)},
        }
        validate_tasks(data)
        write_json(output, data)
        print(f"Prepared protocol v3: original Task A, interior B/C, and {eval_starts} held-out starts: {output}", flush=True)
