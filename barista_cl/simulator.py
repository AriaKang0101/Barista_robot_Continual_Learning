"""ZeroMQ bridge. Loads the exact upstream scene; never saves a .ttt file.

Use a dedicated CoppeliaSim instance. Only joint state/velocity is changed.
Walls, machines, base, camera and original target markers are never relocated.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .core import SCENE_BLOB, git_blob_sha

PATHS = {
    "base": "/Base",
    "j1": "/Base/joint1",
    "l1": "/Base/joint1/link1",
    "j2": "/Base/joint1/link1/joint2",
    "l2": "/Base/joint1/link1/joint2/link2",
    "ee": "/Base/joint1/link1/joint2/link2/End_Effector",
    "camera": "/Vision_Sensor",
    "wall": "/Plane/Wall",
    "machine1": "/Machine1",
    "machine2": "/Machine2",
    "target1": "/FirstTarget",
    "target2": "/EndTarget",
}
FIXED = ["base", "wall", "machine1", "machine2", "camera", "target1", "target2"]
# CoppeliaSim's dynamics initialization can settle a nominally fixed shape by
# less than a millimetre. Keep the guard strict enough to reject an edited or
# dragged workcell while allowing that deterministic solver settling.
GEOMETRY_ATOL = 2e-3


class Simulator:
    def __init__(self, scene, host="localhost", port=23000):
        import zmq
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient

        self.scene = Path(scene).resolve()
        if git_blob_sha(self.scene) != SCENE_BLOB:
            raise ValueError("Scene differs from the pinned upstream .ttt. Use fetch-scene or the unmodified original file.")
        self.client = RemoteAPIClient(host=host, port=port)
        self.client.socket.setsockopt(zmq.RCVTIMEO, 20000)
        self.client.socket.setsockopt(zmq.SNDTIMEO, 20000)
        self.client.socket.setsockopt(zmq.LINGER, 0)
        self.closed = False
        try:
            self.sim = self.client.require("sim")
            self.stop()
            # CoppeliaSim must be on the same computer / shared filesystem.
            self.sim.loadScene(str(self.scene))
            self.handles = {name: self.sim.getObject(path) for name, path in PATHS.items()}
            self.joints = [self.handles[k] for k in ["j1", "j2"]]
            self.parts = [self.handles[k] for k in ["l1", "l2", "ee"]]
            self.obstacles = [self.handles[k] for k in ["wall", "machine1", "machine2"]]
            # Adjacent links intentionally meet at their joint. Only non-adjacent
            # pairs are valid self-collision checks for this 2-DoF mechanism.
            self.self_collision_pairs = [(self.handles["l1"], self.handles["ee"])]
            self.fixed_geometry = self.geometry()
            self.sim.setBoolParam(self.sim.boolparam_realtime_simulation, False)
            self.sim.setStepping(True)
            self.runtime = {
                "coppeliasim_version": self.sim.getInt32Param(self.sim.intparam_program_version),
                "physics_engine": self.sim.getInt32Param(self.sim.intparam_dynamic_engine),
                "simulation_dt": self.sim.getSimulationTimeStep(),
            }
        except Exception:
            self.client.socket.close(linger=0)
            self.client.context.term()
            self.closed = True
            raise

    def geometry(self):
        return {k: list(self.sim.getObjectMatrix(self.handles[k], -1)) for k in FIXED}

    def assert_geometry(self, expected=None):
        expected = self.fixed_geometry if expected is None else expected
        now = self.geometry()
        for name in FIXED:
            delta = float(np.max(np.abs(np.asarray(now[name]) - np.asarray(expected[name]))))
            if delta > GEOMETRY_ATOL:
                raise RuntimeError(
                    f"Fixed scene geometry changed: {name} "
                    f"(max matrix delta {delta:.6g}, allowed {GEOMETRY_ATOL:.6g}). "
                    "Reload the original scene."
                )

    def stop(self):
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()
            deadline = time.monotonic() + 15
            while self.sim.getSimulationState() != self.sim.simulation_stopped:
                if time.monotonic() > deadline:
                    raise TimeoutError("CoppeliaSim failed to stop within 15 s")
                time.sleep(0.02)

    def place(self, q):
        # Used only while stopped: kinematic candidate validation and episode reset.
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            raise RuntimeError("Joint teleportation is only permitted while simulation is stopped")
        for handle, angle in zip(self.joints, q):
            self.sim.setJointPosition(handle, float(angle))
            self.sim.setJointTargetVelocity(handle, 0.0)
            self.sim.setJointTargetForce(handle, 50.0)
        for handle in self.parts + self.joints:
            self.sim.resetDynamicObject(handle)

    def too_close(self, clearance):
        if clearance <= 0:
            return False
        return (any(self.sim.checkDistance(p, o, clearance)[0]
                    for p in self.parts for o in self.obstacles)
                or any(self.sim.checkDistance(a, b, clearance)[0]
                       for a, b in self.self_collision_pairs))

    def self_collision(self):
        return any(self.sim.checkCollision(a, b)[0] for a, b in self.self_collision_pairs)

    def start_at(self, q, clearance=0.0):
        self.stop()
        self.assert_geometry()
        self.place(q)
        if self.collision() or self.too_close(clearance):
            raise RuntimeError("Requested initial pose is not collision/clearance safe")
        self.sim.startSimulation()
        self.sim.step()
        if self.collision() or self.too_close(clearance):
            raise RuntimeError("A prepared initial pose collided after physics initialization; run prepare again")
        self.assert_geometry()

    def joint_positions(self):
        return np.asarray([self.sim.getJointPosition(h) for h in self.joints])

    def joint_velocities(self):
        return np.asarray([self.sim.getJointVelocity(h) for h in self.joints])

    def points(self):
        return np.asarray([self.sim.getObjectPosition(self.handles[k], -1) for k in ["j2", "ee"]])

    def original_target_points(self):
        """The two target positions used by the pinned upstream PPO task."""
        return np.asarray([self.sim.getObjectPosition(self.handles[k], -1)
                           for k in ["target1", "target2"]])

    def collision(self):
        return (any(self.sim.checkCollision(p, o)[0] for p in self.parts for o in self.obstacles)
                or self.self_collision())

    def safe_pose(self, q, clearance=0.01):
        self.place(q)
        if self.collision() or np.any(self.points()[:, 2] < -0.05):
            return False
        if self.too_close(clearance):
            return False
        return True

    def random_safe_start(self, rng, sampler, goals, tolerance, target_q=None,
                          curriculum_fraction=1.0, anchor_indices=None):
        """Start at a new continuous safe pose sampled around the connected safe set."""
        anchors = np.asarray(sampler["anchors"], dtype=float)
        lower = np.asarray(sampler["lower"], dtype=float)
        upper = np.asarray(sampler["upper"], dtype=float)
        jitter = np.asarray(sampler["jitter"], dtype=float)
        clearance = float(sampler["clearance"])
        joint1_exclusion = float(sampler["joint1_exclusion_abs"])
        goals = np.asarray(goals, dtype=float)
        if not 0 < curriculum_fraction <= 1:
            raise ValueError("curriculum_fraction must be in (0, 1]")
        if anchor_indices is not None:
            indices = np.asarray(anchor_indices, dtype=int)
            if (indices.ndim != 1 or len(indices) < 1 or np.any(indices < 0)
                    or np.any(indices >= len(anchors))):
                raise ValueError("Invalid curriculum anchor indices")
            anchors = anchors[indices]
        elif target_q is not None and curriculum_fraction < 1:
            target_q = np.asarray(target_q, dtype=float)
            scale = upper - lower
            distance = np.max(np.abs((anchors - target_q) / scale), axis=1)
            count = max(1, int(np.ceil(len(anchors) * curriculum_fraction)))
            anchors = anchors[np.argsort(distance)[:count]]
        for _ in range(int(sampler["max_attempts"])):
            anchor = anchors[int(rng.integers(len(anchors)))]
            q = np.clip(anchor + rng.uniform(-jitter, jitter), lower, upper)
            if abs(q[0]) < joint1_exclusion:
                continue
            self.stop()
            if not self.safe_pose(q, clearance):
                continue
            points = self.points()
            if any(np.all(np.linalg.norm(points - goal, axis=1) < tolerance) for goal in goals):
                continue
            # The candidate must connect locally to an anchor in the known
            # collision-free component, not merely be safe at one point.
            count = max(2, int(np.ceil(np.max(np.abs(anchor - q)) / np.deg2rad(2))) + 1)
            if not all(self.safe_pose(p, clearance) for p in np.linspace(q, anchor, count)):
                continue
            try:
                self.start_at(q, clearance)
            except RuntimeError:
                self.stop()
                continue
            return q
        raise RuntimeError("Could not sample a safe connected random start; inspect sampler bounds/clearance")

    def joint_limits(self):
        lower = np.deg2rad([-110.0, -90.0])
        upper = np.deg2rad([110.0, 90.0])
        for i, h in enumerate(self.joints):
            cyclic, interval = self.sim.getJointInterval(h)
            if not cyclic:
                lower[i] = max(lower[i], interval[0] + 1e-4)
                upper[i] = min(upper[i], interval[0] + interval[1] - 1e-4)
        if np.any(lower >= upper):
            raise RuntimeError("Scene joint limits do not intersect the configured search range")
        return lower, upper

    def advance(self, normalized_action):
        action = np.clip(np.asarray(normalized_action, dtype=float), -1, 1)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite pair")
        for h, velocity in zip(self.joints, 0.8 * action):
            self.sim.setJointTargetVelocity(h, float(velocity))
        self.sim.step()

    def image(self):
        import cv2
        raw, resolution = self.sim.getVisionSensorImg(self.handles["camera"])
        if not raw:
            raise RuntimeError("Vision sensor returned an empty image; check rendering/vision sensor settings")
        image = np.frombuffer(raw, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        image = cv2.cvtColor(cv2.flip(image, 0), cv2.COLOR_RGB2GRAY)
        return cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)

    def close(self):
        if not self.closed:
            try:
                self.stop()
            finally:
                self.client.socket.close(linger=0)
                self.client.context.term()
                self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
