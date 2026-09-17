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
            if not np.allclose(now[name], expected[name], atol=1e-5, rtol=0):
                raise RuntimeError(f"Fixed scene geometry changed: {name}. Reload the original scene.")

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

    def start_at(self, q):
        self.stop()
        self.assert_geometry()
        self.place(q)
        self.sim.startSimulation()
        self.sim.step()
        if self.collision():
            raise RuntimeError("A prepared initial pose collided after physics initialization; run prepare again")
        self.assert_geometry()

    def joint_positions(self):
        return np.asarray([self.sim.getJointPosition(h) for h in self.joints])

    def points(self):
        return np.asarray([self.sim.getObjectPosition(self.handles[k], -1) for k in ["j2", "ee"]])

    def collision(self):
        return any(self.sim.checkCollision(p, o)[0] for p in self.parts for o in self.obstacles)

    def safe_pose(self, q, clearance=0.01):
        self.place(q)
        if self.collision() or np.any(self.points()[:, 2] < -0.05):
            return False
        if clearance > 0:
            if any(self.sim.checkDistance(p, o, clearance)[0] for p in self.parts for o in self.obstacles):
                return False
        return True

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

