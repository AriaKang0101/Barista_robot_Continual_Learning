"""Run python -m barista_cl --help from the repository root."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import urllib.request

from .core import SCENE_BLOB, SCENE_NAME, UPSTREAM, UPSTREAM_COMMIT, git_blob_sha, read_json


def fetch_scene(destination):
    path = Path(destination)
    if path.exists():
        if git_blob_sha(path) != SCENE_BLOB:
            raise FileExistsError("Destination exists but differs; choose a new path. Nothing overwritten.")
        print(f"Already verified: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{UPSTREAM}/{UPSTREAM_COMMIT}/{SCENE_NAME}"
    tmp = path.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            tmp.write_bytes(response.read())
        if git_blob_sha(tmp) != SCENE_BLOB:
            raise ValueError("Downloaded scene failed the upstream Git blob hash check")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Downloaded exact upstream scene: {path}")


def doctor():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["torch", "stable-baselines3", "gymnasium", "numpy", "coppeliasim-zmqremoteapi-client", "opencv-python-headless"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT INSTALLED"
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError as exc:
        result["torch_import_error"] = str(exc)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Sequential PPO vs policy-Fisher PPO-EWC in the unchanged upstream workcell")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Print versions/GPU; no simulator access")
    fetch = commands.add_parser("fetch-scene", help="Download pinned upstream .ttt without changing it")
    fetch.add_argument("--output", default=f"scenes/{SCENE_NAME}")

    def scene_args(p):
        p.add_argument("--scene", default=f"scenes/{SCENE_NAME}")
        p.add_argument("--host", default="localhost")
        p.add_argument("--port", type=int, default=23000)

    prep = commands.add_parser("prepare", help="Discover goals and dynamically validate all selected start/goal pairs")
    scene_args(prep)
    prep.add_argument("--output", default="artifacts/tasks.json")
    prep.add_argument("--seed", type=int, default=2026)
    prep.add_argument("--grid-size", type=int, default=17)
    prep.add_argument("--tolerance", type=float, default=0.05)
    prep.add_argument("--clearance", type=float, default=0.01)
    prep.add_argument("--train-starts", type=int, default=6)
    prep.add_argument("--eval-starts", type=int, default=3)
    prep.add_argument("--max-steps", type=int, default=500)
    preview = commands.add_parser("preview", help="Show a validated controller route, not a learned PPO result")
    scene_args(preview)
    preview.add_argument("--tasks", default="artifacts/tasks.json")
    preview.add_argument("--task", choices=list("ABC"), default="A")
    train = commands.add_parser("train", help="Run one method/seed; preserves model and optimizer across tasks")
    scene_args(train)
    train.add_argument("--tasks", default="artifacts/tasks.json")
    train.add_argument("--config", default="configs/default.json")
    train.add_argument("--method", choices=["sequential", "ewc"], required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"])
    train.add_argument("--only-task", choices=list("ABC"), help="Train one task from scratch for a learnability pilot")
    compare = commands.add_parser("compare", help="Plot completed paired runs and calculate forgetting/BWT")
    compare.add_argument("runs", nargs="+")
    compare.add_argument("--output", default="artifacts/comparison")
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "fetch-scene":
        fetch_scene(args.output)
    elif args.command == "prepare":
        from .prepare import prepare
        if Path(args.output).exists():
            parser.error("Task file already exists; use another --output to preserve experiment provenance")
        kwargs = vars(args).copy()
        kwargs.pop("command")
        prepare(**kwargs)
    elif args.command == "preview":
        from .core import validate_tasks
        from .prepare import drive_route
        from .simulator import Simulator
        data = read_json(args.tasks)
        validate_tasks(data)
        with Simulator(args.scene, args.host, args.port) as sim:
            sim.assert_geometry(data["fixed_geometry"])
            goal = next(t["points"] for t in data["tasks"] if t["id"] == args.task)
            import numpy as np
            print("This is a validation controller, NOT PPO.")
            print(drive_route(sim, np.array(data["preview_paths"][args.task]), goal,
                              data["goal_tolerance"], data["max_steps"]))
    elif args.command == "train":
        from .experiment import run
        run(args.scene, args.tasks, args.config, args.output, args.method, args.seed, args.device,
            args.host, args.port, args.only_task)
    elif args.command == "compare":
        from .report import compare
        compare(args.runs, args.output)


if __name__ == "__main__":
    main()

