---
title: Selene ML APIs
emoji: 💊
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
short_description: GMM clustering + HistGBM simulation APIs for Selene
---

# Selene ML APIs

Combined FastAPI inference server exposing two models:

| Endpoint | Model | Description |
|---|---|---|
| `POST /api/v1/cluster/predict` | GMM (k=12) | Assigns patient to a WHO-MEC-informed risk cluster |
| `POST /api/v1/simulator/simulate` | HistGBM | Predicts 12-month symptom trajectory per pill |
| `GET /api/v1/health` | — | Combined health check |

## Base URL

```
https://pietrosaveri-selene-ml-apis.hf.space
```

## Quick test

```bash
curl https://pietrosaveri-selene-ml-apis.hf.space/api/v1/health
```
