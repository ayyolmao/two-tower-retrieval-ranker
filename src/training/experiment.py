"""MLflow experiment helpers for the two-tower project.

All training scripts should import from here rather than calling mlflow
directly, so experiment name, tracking URI, and run structure stay consistent.

Usage
-----
    from src.training.experiment import get_experiment, start_run, log_metrics

    get_experiment()   # sets tracking URI + experiment; call once per script
    with start_run("two-tower-v1", params={"embedding_dim": 64}) as run:
        for step, loss in enumerate(train()):
            log_metrics({"train_loss": loss}, step=step)

    # Nested runs (e.g. hyperparameter sweep):
    with start_run("sweep") as outer:
        with start_run("trial-1", nested=True, params={"lr": 1e-3}) as inner:
            log_metrics({"val_loss": 0.4}, step=0)

Smoke test
----------
    python -m src.training.experiment
    mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import mlflow
import yaml

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "mlflow.yaml"

_REQUIRED_KEYS = ("experiment_name", "tracking_uri")


def _load_config(config_path: str | Path = _DEFAULT_CONFIG) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"MLflow config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"MLflow config is empty: {path}")
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"MLflow config {path} missing keys: {missing}")
    return cfg


def _resolve_tracking_uri(uri: str, config_path: Path) -> str:
    """Resolve a relative sqlite:/// path to absolute, anchored at project root."""
    if not uri.startswith("sqlite:///"):
        return uri
    db = Path(uri[len("sqlite:///"):])
    if db.is_absolute():
        return uri
    # config lives at <root>/configs/mlflow.yaml — parent.parent = project root
    return "sqlite:///" + str(config_path.resolve().parent.parent / db)


def get_experiment(config_path: str | Path = _DEFAULT_CONFIG) -> str:
    """Configure MLflow tracking and return the experiment ID.

    Creates the experiment if it does not already exist. Call once per script
    before any start_run() calls.
    """
    config_path = Path(config_path)
    cfg = _load_config(config_path)
    uri = _resolve_tracking_uri(cfg["tracking_uri"], config_path)
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(cfg["experiment_name"])
    return experiment.experiment_id


@contextlib.contextmanager
def start_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
):
    """Context manager that opens an MLflow run, logs params/tags, then closes it.

    Requires get_experiment() to have been called first to set the tracking URI
    and experiment. Yields the active ``mlflow.ActiveRun``.

    Pass ``nested=True`` for inner runs in hyperparameter sweeps or CV loops.
    """
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
        if params:
            mlflow.log_params(params)
        yield run


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of scalar metrics to the active MLflow run.

    '@' in metric names is replaced with '_at_' before logging — MLflow
    rejects '@' in metric keys (e.g. 'recall@100' → 'recall_at_100').
    """
    sanitised = {k.replace("@", "_at_"): v for k, v in metrics.items()}
    mlflow.log_metrics(sanitised, step=step)


# ---------------------------------------------------------------------------
# Smoke test — creates one dummy run to verify the setup end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = _load_config()
    print(f"Experiment : {cfg['experiment_name']}")
    print(f"Tracking UI: mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db\n")

    get_experiment()

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
        for step in range(3):
            log_metrics(
                {
                    "train_loss": round(0.693 - step * 0.05, 4),
                    "recall_at_100": round(step * 0.01, 4),
                },
                step=step,
            )
        print(f"Run ID : {run.info.run_id}")

    print("\nDone. Open the MLflow UI to inspect the run:")
    print("  mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db")
