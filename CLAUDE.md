# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For exact reproducibility, install from the lock file instead:

```bash
pip install -r requirements-lock.txt
```

Verify the ML toolchain:

```bash
python -c "import torch, mlflow, faiss, fastapi; print('OK')"
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"  # should print True on Apple Silicon
```

Launch the MLflow experiment-tracking UI (port 5001 to avoid conflicts):

```bash
mlflow ui --port 5001
# Opens at http://127.0.0.1:5001
```

## MLflow experiment tracking

```bash
# Smoke-test: creates one dummy run in the two-tower-retrieval-ranker experiment
python -m src.training.experiment

# Open the UI to inspect the run (experiment name: two-tower-retrieval-ranker)
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
# → http://127.0.0.1:5001
```

Training scripts should use the helpers in `src/training/experiment.py`:

```python
from src.training.experiment import get_experiment, start_run, log_metrics

get_experiment()   # sets tracking URI + experiment; call once per script
with start_run("my-run", params={"embedding_dim": 64}) as run:
    log_metrics({"train_loss": 0.5}, step=0)

# Nested run (hyperparameter sweep, CV loop):
with start_run("sweep") as outer:
    with start_run("trial-1", nested=True, params={"lr": 1e-3}) as inner:
        log_metrics({"val_loss": 0.4}, step=0)
```

Config is in `configs/mlflow.yaml` (experiment name, tracking URI).

## Data pipeline commands

```bash
# Download flat/50m likes + listens into data/raw/ (multi-GB, idempotent)
python -m src.data.download --size 50m --events likes listens

# Build committed smoke-test sample in data/sample/ (requires data/raw/)
python -m src.data.load --event likes

# Load a parquet file programmatically
python -c "from src.data.load import load_interactions; print(load_interactions('likes').head())"
```

`data/raw/` is gitignored. `data/sample/` holds a 200-user deterministic subset committed to the repo so the pipeline is testable without a full download.

## Architecture

Two-stage recommendation pipeline:

**Stage 1 — Retrieval:** User tower + item tower produce embeddings; dot-product similarity finds top-K candidates via a FAISS ANN index.

**Stage 2 — Ranking:** DCN-v2 or XGBoost LambdaMART re-ranks the top-K candidates. Output is served via `FastAPI /recommendations/{user_id}`.

## Dataset

[Yandex Yambda](https://huggingface.co/datasets/yandex/yambda) — Yandex Music interaction logs. Three scales: `50m` (dev default), `500m`, `5b`. Two layouts: `flat` (used here) and `sequential`. Event types: `likes`, `listens`, `dislikes`, `multi_event`, `unlikes`, `undislikes`.

Flat schema: `uid` (uint32), `item_id` (uint32), `timestamp` (uint32, 5s bins), `is_organic` (uint8). Listens also have `played_ratio_pct` and `track_length_seconds`.

## Config

`configs/data.yaml` controls dataset size, events, raw/sample paths, and sample parameters (`n_users`, `seed`). The loaders in `src/data/` use these values as defaults but accept CLI overrides.

## Project status

Phase 0 complete. Phase 1 in progress: architecture outline and MLflow experiment helper are done; metrics, baselines, and model code not started yet.

## Evaluation targets

Recall@100, NDCG@10, MRR@10 (full pipeline). Baselines: popularity and matrix factorization. Splits: time-based. Ablations: embedding dim (32/64/128), negative sampling strategy, user-history length.
