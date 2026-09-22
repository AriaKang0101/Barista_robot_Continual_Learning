import copy
import csv
import shutil

import numpy as np
import pytest
import torch

from barista_cl.core import read_json, validate_tasks, write_json
from conftest import FakeSimulator


def test_automatic_goal_preparation(monkeypatch, tmp_path):
    import barista_cl.prepare as module
    monkeypatch.setattr(module, "Simulator", FakeSimulator)
    path = tmp_path / "tasks.json"
    module.prepare("unused.ttt", path, grid_size=7, eval_starts=6)
    tasks = read_json(path)
    validate_tasks(tasks)
    assert tasks["schema_version"] == 4
    assert tasks["protocol"] == "reachable_state_cl_v4"
    assert len(tasks["tasks"]) == 3
    assert all(len(tasks["eval_starts"][task]) == 6 for task in "ABC")
    assert len(tasks["training_sampler"]["anchors"]) >= 3


def test_failed_dynamic_validation_writes_no_tasks(monkeypatch, tmp_path):
    import barista_cl.prepare as module
    monkeypatch.setattr(module, "Simulator", FakeSimulator)
    monkeypatch.setattr(module, "drive_route", lambda *a, **k: (False, 3, "collision"))
    output = tmp_path / "tasks.json"
    with pytest.raises(RuntimeError, match="Insufficient"):
        module.prepare("unused.ttt", output, grid_size=5, eval_starts=6)
    assert not output.exists()


def test_full_runner_two_methods_three_stages_and_reports(monkeypatch, tmp_path, task_data):
    import barista_cl.experiment as experiment
    from barista_cl.ewc import EWCPPO
    from barista_cl.report import compare
    monkeypatch.setattr(experiment, "Simulator", FakeSimulator)
    tasks_path, config_path = tmp_path / "tasks.json", tmp_path / "config.json"
    write_json(tasks_path, task_data)
    config = read_json("configs/default.json")
    config.update(n_steps=8, batch_size=4, n_epochs=1, timesteps_per_task=8,
                  fisher_samples=4, device="cpu")
    write_json(config_path, config)
    paths = []
    for method in ["sequential", "ewc"]:
        path = tmp_path / method
        experiment.run("unused.ttt", tasks_path, config_path, path, method)
        manifest = read_json(path / "manifest.json")
        assert manifest["status"] == "complete"
        assert manifest["actual_training_steps"] == 24
        assert len(list(path.glob("stage_*.zip"))) == 3
        paths.append(path)
        with (path / "evaluation_episodes.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == (1 + 2 + 3) * len(task_data["eval_starts"]["A"])
        assert {r["task"] for r in rows if r["stage"] == "3"} == set("ABC")
    # Consolidation and evaluation must not alter first-stage PPO parameters.
    a = EWCPPO.load(paths[0] / "stage_1_A.zip", device="cpu")
    b = EWCPPO.load(paths[1] / "stage_1_A.zip", device="cpu")
    for k, v in a.policy.state_dict().items():
        torch.testing.assert_close(v, b.policy.state_dict()[k], rtol=0, atol=0)
    c = EWCPPO.load(paths[1] / "stage_3_C.zip", device="cpu")
    assert len(c.ewc_terms) == 3

    # A failed third stage can be restarted from the committed B boundary in a
    # fresh output without retaining partial stage-C log rows.
    failed = tmp_path / "failed_ewc"
    shutil.copytree(paths[1], failed)
    (failed / "stage_3_C.zip").unlink()
    failed_manifest = read_json(failed / "manifest.json")
    failed_manifest.update(status="failed", completed_stages=2,
                           actual_training_steps=16, error="ZMQError: test disconnect")
    write_json(failed / "manifest.json", failed_manifest)
    failed_matrix = np.loadtxt(failed / "success_matrix.csv", delimiter=",",
                               skiprows=1, ndmin=2)
    failed_matrix[2, :] = np.nan
    np.savetxt(failed / "success_matrix.csv", failed_matrix, delimiter=",",
               header="A,B,C", comments="")
    resumed = tmp_path / "resumed_ewc"
    experiment.run("unused.ttt", tasks_path, config_path, resumed, "ewc",
                   resume_from=failed)
    resumed_manifest = read_json(resumed / "manifest.json")
    assert resumed_manifest["status"] == "complete"
    assert resumed_manifest["completed_stages"] == 3
    assert resumed_manifest["actual_training_steps"] == 24
    assert resumed_manifest["resume_checkpoint"] == "stage_2_B.zip"
    assert len(EWCPPO.load(resumed / "stage_3_C.zip", device="cpu").ewc_terms) == 3
    with (resumed / "evaluation_episodes.csv").open() as f:
        resumed_evaluations = list(csv.DictReader(f))
    assert len(resumed_evaluations) == (1 + 2 + 3) * len(task_data["eval_starts"]["A"])
    with (resumed / "training_episodes.csv").open() as f:
        resumed_training = list(csv.DictReader(f))
    assert {row["stage"] for row in resumed_training} == {"1", "2", "3"}

    compare(paths, tmp_path / "report")
    assert (tmp_path / "report" / "retention.png").exists()
    assert read_json(tmp_path / "report" / "summary.json")["ewc"]["n_seeds"] == 1
    with pytest.raises(FileExistsError):
        experiment.run("unused.ttt", tasks_path, config_path, paths[0], "sequential")


def test_report_rejects_unpaired_seeds(tmp_path):
    from barista_cl.report import compare
    with pytest.raises(ValueError, match="matching"):
        compare([], tmp_path / "report")
