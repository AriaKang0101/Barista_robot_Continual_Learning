"""Sequential subprocesses: one CoppeliaSim server cannot serve concurrent runs."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scenes/safety_rl_2dof.ttt")
    parser.add_argument("--tasks", default="artifacts/tasks_v4.json")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output", default="runs/comparison")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    paths = []
    for seed in args.seeds:
        for method in ["sequential", "ewc"]:
            output = Path(args.output) / f"{method}_seed{seed}"
            paths.append(str(output))
            subprocess.run([sys.executable, "-m", "barista_cl", "train", "--method", method,
                            "--seed", str(seed), "--scene", args.scene, "--tasks", args.tasks,
                            "--config", args.config, "--output", str(output), "--device", args.device,
                            "--host", args.host, "--port", str(args.port)], check=True)
    subprocess.run([sys.executable, "-m", "barista_cl", "compare", *paths,
                    "--output", str(Path(args.output) / "report")], check=True)


if __name__ == "__main__":
    main()
