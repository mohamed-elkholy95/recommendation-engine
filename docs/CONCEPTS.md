# Recommendation System Concepts

A glossary of key terms and concepts used throughout this project.

## Filtering Approaches

| Term | Definition |
|------|-----------|
| **Collaborative Filtering (CF)** | Recommends based on user–item interaction patterns. "Users who liked X also liked Y." |
| **Content-Based Filtering** | Recommends items similar to what the user liked, using item metadata (genres, descriptions). |
| **Hybrid Filtering** | Combines multiple approaches (CF + content + neural) to leverage each method's strengths. |
| **Neural Collaborative Filtering (NCF)** | Uses neural networks instead of dot products to model user–item interactions. |

## Matrix Factorization

| Term | Definition |
|------|-----------|
| **User-Item Matrix** | Sparse matrix R where R[u,i] = rating user u gave item i. Most entries are missing (unrated). |
| **Latent Factors** | Hidden dimensions that explain rating patterns. E.g., a latent factor might capture "preference for action movies." |
| **SVD (Singular Value Decomposition)** | Factorizes R ≈ U·Σ·Vᵀ to discover latent factors. Truncated SVD keeps only the top-k factors. |
| **Embedding** | A dense vector representation of a user or item in latent space. Learned end-to-end in neural approaches. |

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **NDCG@K** | Quality of ranking — rewards relevant items appearing higher in the list. |
| **MAP@K** | Precision averaged at each relevant item's position — sensitive to ranking order. |
| **Precision@K** | Fraction of top-K items that are relevant. |
| **Recall@K** | Fraction of relevant items that appear in top-K. |
| **Hit Rate@K** | Did at least one relevant item appear in top-K? (Binary per user.) |
| **MRR** | Average 1/rank of the first relevant item — "how quickly do users find something good?" |
| **Catalog Coverage** | Fraction of all items recommended to at least one user. Low = popularity bias. |
| **Intra-List Diversity** | Average pairwise distance within recommendation lists. High = diverse recommendations. |
| **Novelty** | Average self-information (−log₂(popularity)) — measures how surprising recommendations are. |

## Common Problems

| Problem | Description | Our Solution |
|---------|-------------|-------------|
| **Cold Start** | New users/items have no interaction history. | Content-based fallback + popularity baseline. |
| **Sparsity** | Most users rate very few items (>95% missing). | Matrix factorization compresses sparse data into dense latent factors. |
| **Popularity Bias** | System over-recommends popular items. | Novelty metrics + MMR diversity re-ranking. |
| **Filter Bubble** | Content-based systems keep recommending similar items. | Hybrid approach + diversity constraints. |
| **Scalability** | Computing similarities for millions of users/items. | Approximate nearest neighbors, sparse matrix operations. |

## Training Concepts

| Term | Definition |
|------|-----------|
| **Implicit Feedback** | Binary signal — did the user interact? (vs. explicit ratings). |
| **Negative Sampling** | Generating synthetic non-interaction pairs for training. Without it, the model never sees "wrong" answers. |
| **Temporal Split** | Train on past, test on future — respects time ordering. Random splits leak future information. |
| **k-Core Filtering** | Remove users with < k ratings AND items with < k ratings. Ensures sufficient signal for learning. |
