# HippoRAG 2 attribution

The retrieval flow and the `run_personalized_pagerank` interface in this folder
are adapted from HippoRAG 2:

- Repository: https://github.com/OSU-NLP-Group/HippoRAG
- Upstream revision inspected: `1438aba`
- Upstream file: `src/hipporag/HippoRAG.py`
- Copyright: 2025 OSU Natural Language Processing
- License: MIT

The upstream flow is fact retrieval, LLM recognition filtering, mixed entity and
passage restart weights, weighted Personalized PageRank, and passage ranking.
This adaptation replaces upstream OpenIE and local storage with the already
materialized PostgreSQL graph and Milvus indexes in this project. The PPR
iteration uses SciPy instead of `igraph.personalized_pagerank` so no second
in-memory graph dependency is required.

The full MIT notice is reproduced in `LICENSE.hipporag`.
