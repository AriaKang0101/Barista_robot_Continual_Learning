import numpy as np
import pytest
import torch

from barista_cl.core import SCENE_BLOB
from barista_cl.simulator import FIXED


@pytest.fixture(autouse=True)
def single_thread_torch():
    torch.set_num_threads(1)


class FakeSimulator:
    """Fast test double only; this is NOT a substitute for CoppeliaSim validation."""
    def __init__(self, *args, **kwargs):
        self.q = np.zeros(2)
        self.hit = False
        self.runtime = {"test_double": True}
        identity = np.eye(4)[:3].ravel().tolist()
        self.fixed_geometry = {name: identity for name in FIXED}
        self.closed = False

    def points(self):
        q1, q2 = self.q
        return np.array([[np.cos(q1), np.sin(q1), 0.2],
                         [np.cos(q1) + np.cos(q1 + q2), np.sin(q1) + np.sin(q1 + q2), 0.2]])

    def original_target_points(self):
        # Exactly represented by the coarse 5x5 and 7x7 preparation grids.
        q1, q2 = 0.0, 0.0
        return np.array([[np.cos(q1), np.sin(q1), 0.2],
                         [np.cos(q1) + np.cos(q1 + q2),
                          np.sin(q1) + np.sin(q1 + q2), 0.2]])

    def start_at(self, q, clearance=0.0):
        self.q = np.array(q, dtype=float)

    def random_safe_start(self, rng, sampler, goals, tolerance, target_q=None,
                          curriculum_fraction=1.0, anchor_indices=None):
        self.last_target_q = None if target_q is None else np.asarray(target_q).copy()
        self.last_curriculum_fraction = curriculum_fraction
        self.last_anchor_indices = None if anchor_indices is None else list(anchor_indices)
        anchors = np.asarray(sampler["anchors"])
        if anchor_indices is not None:
            anchors = anchors[np.asarray(anchor_indices, dtype=int)]
        elif target_q is not None and curriculum_fraction < 1:
            distances = np.max(np.abs(anchors - np.asarray(target_q)), axis=1)
            count = max(1, int(np.ceil(len(anchors) * curriculum_fraction)))
            anchors = anchors[np.argsort(distances)[:count]]
        anchor = anchors[int(rng.integers(len(anchors)))]
        self.q = anchor + rng.uniform(-0.01, 0.01, size=2)
        return self.q.copy()

    def joint_positions(self):
        return self.q.copy()

    def joint_velocities(self):
        return np.zeros(2, dtype=float)

    def advance(self, action):
        self.q += np.clip(action, -1, 1) * 0.04

    def collision(self):
        return self.hit

    def image(self):
        return np.full((84, 84), int((self.q[0] + 4) * 20) % 256, dtype=np.uint8)

    def assert_geometry(self, expected=None):
        pass

    def joint_limits(self):
        return np.array([-1.2, -1.2]), np.array([1.2, 1.2])

    def safe_pose(self, q, clearance=0.01):
        self.q = np.array(q)
        return True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@pytest.fixture
def fake_sim():
    return FakeSimulator()


@pytest.fixture
def task_data(fake_sim):
    tasks = []
    for label, q in zip("ABC", [[0.9, 0.2], [-0.9, 0.2], [0.1, -1.1]]):
        fake_sim.start_at(q)
        tasks.append({"id": label, "q": q, "points": fake_sim.points().tolist()})
    anchors = [[-0.8, -0.4], [0.0, 0.0], [0.8, 0.4]]
    eval_starts = {
        "A": [[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2]],
        "B": [[0.4, 0.4], [0.5, 0.4], [0.4, 0.5]],
        "C": [[-0.5, -0.5], [-0.4, -0.5], [-0.5, -0.4]],
    }
    return {"schema_version": 4, "protocol": "reachable_state_cl_v4",
            "scene_blob": SCENE_BLOB, "runtime": fake_sim.runtime,
            "fixed_geometry": fake_sim.fixed_geometry, "tasks": tasks,
            "goal_tolerance": 0.05, "max_steps": 5,
            "training_sampler": {"kind": "task_conditioned_graph_curriculum", "anchors": anchors,
                "lower": [-1.2, -1.2], "upper": [1.2, 1.2], "jitter": [0.1, 0.1],
                "clearance": 0.01, "joint1_exclusion_abs": 0.0, "max_attempts": 20,
                "task_anchor_order": {label: [0, 1, 2] for label in "ABC"}},
            "eval_starts": eval_starts,
            "eval_bands": {label: ["near", "medium", "far"] for label in "ABC"},
            "dynamic_validation": {"all_passed": True}}
