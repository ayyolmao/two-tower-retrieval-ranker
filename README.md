# Two-Tower Retrieval + Ranker

A complete **two-stage recommender** built on a large open dataset: a **two-tower
retrieval** model (user tower, item tower, dot-product scoring, ANN index for
serving) followed by a **DCN-v2 / LambdaMART ranker** that re-ranks the top-K
candidates. The aim is a faithful, miniature version of the production ranking
architecture used at Meta, Pinterest, and Spotify — with reproducible offline
metrics and an online-serving demo.

> **Status:** 🚧 Active build. Repo bootstrap + data pipeline are in place; modeling
> is in progress (see [Roadmap](#roadmap)).

## Why this project

Production recommendation systems are overwhelmingly two-stage: a cheap
**retrieval** model narrows millions of items to a few hundred candidates, then an
expensive **ranking** model orders them. This repo implements that full pipeline
end-to-end so the design choices that matter in industry — negative sampling,
ANN candidate generation, multi-stage evaluation, serving latency — are made
explicit and measured.

## Architecture

```
                    ┌─────────────┐        ┌─────────────┐
   user features ──▶│  User Tower │        │  Item Tower │◀── item features
                    └──────┬──────┘        └──────┬──────┘
                           │   dot product (sim)  │
                           └──────────┬───────────┘
                                      ▼
                        ┌──────────────────────────┐
                        │   FAISS ANN index (top-K) │   ← Stage 1: Retrieval
                        └─────────────┬─────────────┘
                                      ▼  top-K candidates
                        ┌──────────────────────────┐
                        │  DCN-v2 / LambdaMART      │   ← Stage 2: Ranking
                        │  re-ranker                │
                        └─────────────┬─────────────┘
                                      ▼
                            ranked recommendations
                     (served via FastAPI /recommendations/{user_id})
```

## Dataset

**[Yandex Yambda](https://huggingface.co/datasets/yandex/yambda)** — Yandex Music
interaction logs, released May 2025 as "the world's largest open dataset for
recommender systems." Available in **50M / 500M / 5B** scales (Parquet, Apache 2.0).
This project develops on the **50M** flavor (10K users, 47.6M interactions) and is
parametrized to scale up later.

**Flat schema:** `uid`, `item_id`, `timestamp` (5s bins), `is_organic`, plus
`played_ratio_pct` and `track_length_seconds` for listens. Audio embeddings are
also provided.

### Download

```bash
pip install -r requirements.txt

# Download flat/50m likes + listens into data/raw/ (multi-GB; idempotent)
python -m src.data.download --size 50m --events likes listens

# Build a tiny committed smoke-test sample (no full download needed if data/raw exists)
python -m src.data.load --event likes
```

`data/raw/` is gitignored; a tiny user-sampled subset lives in `data/sample/` so the
pipeline is runnable without the large download.

## Planned evaluation

- **Metrics:** Recall@K, NDCG@K, MRR@K (full-pipeline: Recall@100, NDCG@10, MRR@10) + serving p95 latency.
- **Baselines:** popularity, matrix factorization.
- **Splits:** time-based train / validation / test.
- **Ablations:** embedding dim (32 / 64 / 128), negative sampling (random vs popularity-weighted vs in-batch), user-history length.

## Roadmap

This checklist mirrors the project's phased task plan and doubles as a progress tracker.

### Phase 0 — Setup
- [x] Create GitHub repo
- [x] Write README + project goal
- [x] Dataset download / load scripts
- [ ] Set up tooling (PyTorch, MLflow, Docker, FastAPI, FAISS)
- [ ] Read *Deep Neural Networks for YouTube Recommendations* + *Sampling-Bias-Corrected Neural Two-Tower* papers
- [ ] Skim Eugene Yan — *Patterns for Personalization*

### Phase 1 — Two-Tower Retrieval + Ranker
- [ ] Architecture outline + MLflow experiment setup
- [ ] Metrics: Recall@K, NDCG@K, MRR@K
- [ ] Popularity + matrix-factorization baselines
- [ ] Time-based train / validation / test split
- [ ] Negative sampling (random + popularity-weighted)
- [ ] User tower + item tower with in-batch negatives
- [ ] Ablations: embedding dim, negative sampling, history length
- [ ] Export item embeddings + build FAISS index
- [ ] Ranker (XGBoost LambdaMART or DCN-v2)
- [ ] Full-pipeline eval (Recall@100, NDCG@10, MRR@10, p95 latency)
- [ ] FastAPI `/recommendations/{user_id}` endpoint
- [ ] Dockerfile + one-command local run
- [ ] Blog post #1: *Two-Stage Recommender from Scratch*

## Repository structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── data.yaml          # dataset size, events, paths, sample config
├── src/
│   └── data/
│       ├── download.py     # download Yambda subsets from HF Hub
│       └── load.py         # load parquet + build smoke-test sample
├── data/                   # downloads gitignored; data/sample/ committed
└── notebooks/              # EDA (coming soon)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License

Code: MIT. The Yambda dataset is licensed by Yandex under Apache 2.0 and is **not**
redistributed here — use the download script to fetch it from the Hugging Face Hub.
