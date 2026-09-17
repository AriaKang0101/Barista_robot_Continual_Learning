"""Compare complete, matched runs. Error bars are across training seeds."""
import csv
from pathlib import Path

import numpy as np

from .core import read_json, write_json


def compare(paths, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {"sequential": {}, "ewc": {}}
    reference = None
    rows = []
    for path in map(Path, paths):
        manifest, config = read_json(path / "manifest.json"), read_json(path / "config.json")
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete run: {path}")
        identity = {"tasks": manifest["tasks_sha256"], "order": manifest["task_order"],
                    "config": {k: v for k, v in config.items() if k != "seed"},
                    "device": manifest["resolved_device"], "versions": manifest["versions"],
                    "steps": manifest["actual_training_steps"]}
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"Incompatible task/config/runtime/budget: {path}")
        method, seed = manifest["method"], manifest["seed"]
        if seed in groups[method]:
            raise ValueError(f"Duplicate {method} seed={seed}")
        with (path / "evaluation_episodes.csv").open(encoding="utf-8") as f:
            evaluations = list(csv.DictReader(f))
        n = len(manifest["task_order"])
        final = [r for r in evaluations if int(r["stage"]) == n]
        summary = read_json(path / "metrics.json")
        summary.update({"final_mean_collision": float(np.mean([float(r["collision"]) for r in final])),
                        "final_mean_timeout": float(np.mean([float(r["timeout"]) for r in final])),
                        "elapsed_seconds": manifest["elapsed_seconds"]})
        groups[method][seed] = (path, summary)
        rows.append({"method": method, "seed": seed, **summary})
    if not groups["sequential"] or set(groups["sequential"]) != set(groups["ewc"]):
        raise ValueError("Supply both methods with exactly matching training seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregated = {}
    names = ["final_mean_success", "mean_forgetting", "backward_transfer", "final_mean_collision", "final_mean_timeout"]
    for method, seeds in groups.items():
        aggregated[method] = {"n_seeds": len(seeds), "metrics": {}}
        for name in names:
            values = [summary[name] for _, summary in seeds.values()]
            aggregated[method]["metrics"][name] = {"mean": float(np.mean(values)),
                "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    write_json(output / "summary.json", aggregated)
    order = reference["order"]
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4), squeeze=False)
    for method, seeds in groups.items():
        matrices = np.stack([np.loadtxt(path / "success_matrix.csv", delimiter=",", skiprows=1, ndmin=2)
                             for path, _ in seeds.values()])
        for j, task in enumerate(order):
            values = matrices[:, j:, j]
            x = np.arange(j + 1, len(order) + 1)
            mean = np.mean(values, axis=0)
            ax = axes[0, j]
            ax.plot(x, mean, marker="o", label=method)
            if len(seeds) > 1:
                sd = np.std(values, axis=0, ddof=1)
                ax.fill_between(x, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), alpha=0.15)
            ax.set(title=f"Task {task}", xlabel="Completed training stage", ylabel="Success rate", ylim=(-0.02, 1.02))
            ax.set_xticks(range(1, len(order) + 1))
            ax.legend()
            ax.grid(alpha=0.25)
    fig.suptitle("Retention on held-out starts (bands: SD across training seeds)")
    fig.tight_layout()
    fig.savefig(output / "retention.png", dpi=180)
    plt.close(fig)
    print(f"Comparison written to {output}; seeds per method: {len(groups['ewc'])}")

