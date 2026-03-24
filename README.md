<div align="center">

# 🎯 Recommendation Engine

**Hybrid recommendation system** with collaborative filtering, content-based filtering, and matrix factorization

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-39%20passed-success?style=flat-square)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)

</div>

## Overview

A **hybrid recommendation engine** combining collaborative filtering (user-item interactions), content-based filtering (item feature similarity), and matrix factorization (latent factor decomposition). Includes diverse recommendation strategies and comprehensive evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Hybrid Ensemble                       │
│         (weighted fusion + MMR diversity)                │
├──────────────┬──────────────────┬───────────────────────┤
│  Collaborative │  Content-Based  │   Neural CF (NCF)    │
│  Filtering (CF)│   Filtering     │                      │
│  ─────────────│  ──────────────  │  ───────────────     │
│  SVD matrix   │  TF-IDF + genre  │  GMF + MLP branches  │
│  factorization│  multi-hot       │  with negative        │
│               │  encoding        │  sampling             │
├──────────────┴──────────────────┴───────────────────────┤
│              Data Preprocessing Layer                    │
│   k-core filtering · temporal train/test split          │
│   interaction matrix · data validation                  │
├─────────────────────────────────────────────────────────┤
│              Evaluation Suite                            │
│   NDCG · MAP · Precision · Recall · MRR · Hit Rate      │
│   Catalog Coverage · Intra-List Diversity · Novelty      │
└─────────────────────────────────────────────────────────┘
```

## Features

- 👥 **Collaborative Filtering** — SVD-based matrix factorization with user-mean centering
- 📄 **Content-Based Filtering** — TF-IDF + genre features with explainable recommendations
- 🧠 **Neural Collaborative Filtering** — GMF + MLP architecture with negative sampling
- 🔀 **Hybrid Ensemble** — Weighted score fusion with MMR diversity re-ranking
- 🎲 **Cold Start Handling** — Content-based fallback + popularity baseline
- 📊 **9 Evaluation Metrics** — NDCG, MAP, Precision, Recall, MRR, Hit Rate, Coverage, Diversity, Novelty
- 🔍 **Data Validation** — Automated quality checks before training
- 🌐 **REST API** — FastAPI with request logging and error handling
- ✅ **50+ Tests** — Edge cases, metric validation, and data quality checks

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/recommendation-engine.git
cd recommendation-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start API server
python -m src.api.main

# Launch dashboard
streamlit run streamlit_app/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and model status |
| GET | `/recommendations/{user_id}?n=10` | Personalized top-N recommendations |
| GET | `/similar/{item_id}?n=10` | Content-similar movies |
| GET | `/trending?n=10` | Trending/popular movies |
| POST | `/rate` | Submit a user rating |

## Project Structure

```
├── src/
│   ├── models/
│   │   ├── collaborative.py    # SVD matrix factorization
│   │   ├── content_based.py    # TF-IDF + genre features
│   │   ├── neural_cf.py        # PyTorch NCF with negative sampling
│   │   └── hybrid.py           # Weighted ensemble + MMR
│   ├── data/
│   │   ├── loader.py           # MovieLens loading + synthetic generation
│   │   └── preprocessor.py     # Cleaning, splitting, validation
│   ├── evaluation.py           # 9 recommendation metrics
│   ├── api/main.py             # FastAPI REST service
│   └── config.py               # Hyperparameters and paths
├── tests/                      # 50+ pytest test cases
├── docs/
│   └── CONCEPTS.md             # Glossary of recommendation terms
└── requirements.txt
```

## Key Design Decisions

1. **Temporal train/test split** instead of random — prevents data leakage from future interactions
2. **Negative sampling** for NCF — teaches the model what users *don't* want, not just what they do
3. **MMR re-ranking** — prevents recommendation lists from being too homogeneous
4. **Data validation before training** — catches null values, duplicates, and range errors early

## Further Reading

- See [docs/CONCEPTS.md](docs/CONCEPTS.md) for a glossary of recommendation system terms
- Each source module contains educational docstrings with theory explanations and references

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
