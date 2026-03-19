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

## Features

- 👥 **Collaborative Filtering** — User-based and item-based similarity
- 📄 **Content-Based Filtering** — Feature similarity with TF-IDF representations
- 🧮 **Matrix Factorization** — SVD-based latent factor decomposition
- 🎲 **Cold Start Handling** — Popularity-based recommendations for new users
- 🔄 **Diversity Scoring** — Intra-list diversity metrics
- 📊 **5-Page Dashboard** — Discover, profile, training, metrics, and analysis views
- ✅ **39 Tests** — Full coverage of all recommendation strategies

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/recommendation-engine.git
cd recommendation-engine
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
