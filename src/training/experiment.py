"""MLflow experiment helpers for the two-tower project.

All training scripts should import from here rather than calling mlflow
directly, so experiment name, tracking URI, and run structure stay consistent.

Usage
-----
    from src.training.experiment import get_experiment, start_run, log_metrics

    get_experiment()          # idempotent; call once at the top of a training script
    with start_run("two-tower-v1", params={"embedding_dim": 64}) as run:
        for step, loss in enumerate(train()):
            log_metrics({"train_loss": loss}, step=step)

Smoke test
----------
    python -m src.training.experiment
    mlflow ui --port 5001
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import mlflow
import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "mlflow.yaml"


def _load_config(config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_experiment(config_path: str | Path = _DEFAULT_CONFIG) -> str:
    """Configure MLflow tracking and return the experiment ID.

    Creates the experiment if it does not already exist. Safe to call multiple
    times (mlflow.set_experiment is idempotent).
    """
    cfg = _load_config(config_path)
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    experiment = mlflow.set_experiment(cfg["experiment_name"])
    return experiment.experiment_id


@contextlib.contextmanager
def start_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    config_path: str | Path = _DEFAULT_CONFIG,
):
    """Context manager that opens an MLflow run, logs params/tags, then closes it.

    Yields the active ``mlflow.ActiveRun`` so callers can access ``run.info``.
    """
    get_experiment(config_path)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        if params:
            mlflow.log_params(params)
        yield run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of scalar metrics to the active MLflow run."""
    mlflow.log_metrics(metrics, step=step)


# ---------------------------------------------------------------------------
# Smoke test — creates one dummy run to verify the setup end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = _load_config()
    print(f"Experiment : {cfg['experiment_name']}")
    print(f"Tracking UI: mlflow ui --port 5001 --backend-store-uri {cfg['tracking_uri']}\n")

    with start_run(
        "smoke-test",
        params={
            "embedding_dim": 64,
            "neg_samples": 100,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "model": "two-tower",
        },
        tags={"stage": "smoke", "dataset": "yambda-50m"},
    ) as run:
        # Simulate a few training steps
        for step in range(3):
            log_metrics(
                {
                    "train_loss": round(0.693 - step * 0.05, 4),
                    "recall_at_100": round(step * 0.01, 4),
                },
                step=step,
            )
        print(f"Run ID : {run.info.run_id}")
        print(f"Status : {run.info.status}")

    print("\nDone. Open the MLflow UI to inspect the run:")
    print("  mlflow ui --port 5001")
