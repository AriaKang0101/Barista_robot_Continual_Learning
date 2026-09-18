import copy
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from barista_cl.core import continual_metrics, largest_component, reached, route, validate_tasks
from barista_cl.env import BaristaEnv


def test_joint_success_is_and_and_strict():
    goal = [[0, 0, 0], [1, 0, 0]]
    assert not reached([[0, 0, 0], [1.2, 0, 0]], goal, 0.1)
    assert not reached([[0.1, 0, 0], [1, 0, 0]], goal, 0.1)
    assert reached([[0, 0, 0], [1.01, 0, 0]], goal, 0.1)


def test_forgetting_and_positive_transfer():
    r = [[0.9, np.nan, np.nan], [0.7, 0.8, np.nan], [0.6, 0.9, 1.0]]
    m = continual_metrics(r)
    assert m["mean_forgetting"] == pytest.approx(0.1)
    assert m["backward_transfer"] == pytest.approx(-0.1)
    assert m["final_mean_success"] == pytest.approx(2.5 / 3)
    assert continual_metrics([[0.7]])["mean_forgetting"] == 0


def test_graph_excludes_disconnected_nodes():
    graph = {0: {1}, 1: {0, 2}, 2: {1}, 4: set()}
    assert largest_component(graph) == [0, 1, 2]
    assert route(graph, 0, 2) == [0, 1, 2]
    with pytest.raises(ValueError):
        route(graph, 0, 4)


def test_invalid_goals_and_overlap_rejected(task_data):
    validate_tasks(task_data)
    bad = copy.deepcopy(task_data)
    bad["tasks"][1]["points"] = bad["tasks"][0]["points"]
    with pytest.raises(ValueError, match="Overlapping"):
        validate_tasks(bad)
    bad = copy.deepcopy(task_data)
    bad["dynamic_validation"]["all_passed"] = False
    with pytest.raises(ValueError, match="prepare"):
        validate_tasks(bad)
    bad = copy.deepcopy(task_data)
    bad["eval_starts"][1] = bad["eval_starts"][0]
    with pytest.raises(ValueError, match="distinct"):
        validate_tasks(bad)


def test_environment_seed_frame_reset_and_checker(fake_sim, task_data):
    env = BaristaEnv(fake_sim, task_data)
    env.set_curriculum_fraction(0.25)
    check_env(env, skip_render_check=True)
    a, ai = env.reset(seed=77)
    env.step(np.ones(2))
    b, bi = env.reset(seed=77)
    assert ai["task_id"] == bi["task_id"] and ai["start_index"] == bi["start_index"]
    np.testing.assert_allclose(ai["start_q"], bi["start_q"])
    np.testing.assert_array_equal(a["image"], b["image"])
    assert b["goal"].shape == (6,)
    assert b["joints"].shape == (2,)
    assert np.all(np.abs(b["joints"]) <= 1)
    assert fake_sim.last_curriculum_fraction == pytest.approx(0.25)
    np.testing.assert_allclose(fake_sim.last_target_q, task_data["tasks"][0]["q"])
    np.testing.assert_array_equal(b["image"][0], b["image"][-1])


def test_collision_takes_priority_over_success(fake_sim, task_data):
    env = BaristaEnv(fake_sim, task_data)
    env.reset(seed=0)
    fake_sim.q = np.array(task_data["tasks"][0]["q"])
    fake_sim.hit = True
    _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated and not truncated and info["collision"]
    assert not info["is_success"] and info["done_reason"] == "collision"
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(2))


def test_timeout_and_task_switch(fake_sim, task_data):
    env = BaristaEnv(fake_sim, task_data)
    obs_a, _ = env.reset(seed=0)
    for _ in range(task_data["max_steps"]):
        _, _, terminated, truncated, info = env.step(np.zeros(2))
    assert not terminated and truncated and info["done_reason"] == "timeout"
    env.set_task("B")
    obs_b, _ = env.reset(seed=0)
    assert not np.array_equal(obs_a["goal"], obs_b["goal"])
