# ecommerce-multimodal-rag

Multimodal conversational AI for e-commerce: CLIP + FAISS retrieval and RAG over the Amazon Product Dataset 2020.

## Setup

```bash
# Recommended: conda base (or any env with the packages below)
conda activate base

cd ecommerce-multimodal-rag
pip install -r requirements.txt

# API key for LLM / RAG (later branches)
cp .env.example .env
# edit .env and set OPENAI_API_KEY=...
```

## Artifact policy

| Path | In git? | Notes |
|------|---------|--------|
| `data/processed/clean_products.csv` | **Yes** | Clean ~10k-product catalog for demos |
| `data/raw/` | No | Original Kaggle dump; download separately if needed |
| `data/processed/*.npy` | No | CLIP embeddings (rebuild with `src/clip_embeddings.py`) |
| `data/processed/*.index` | No | FAISS index (rebuild with `src/build_faiss.py`) |
| `data/images/` | No | Product JPEGs (download with image scripts) |
| `.env` | No | Secrets only — use `.env.example` as a template |

If embeddings or the FAISS index are missing locally, regenerate them from the clean CSV before running search.

## Current scripts (baseline)

Run from the repo root (or adjust paths):

- `src/clip_embeddings.py` — build text embeddings
- `src/build_faiss.py` — build FAISS index
- `src/search_products.py` — text retrieval demo
- `python -m src.evaluate_retrieval --sample-size 100 --top-k 10` — Recall@1/5/10
- `src/download_images.py` / `src/image_embeddings.py` — image pilot (10 products)

Project scope and branch plan: [`docs/ASSIGNMENT_PLAN.md`](docs/ASSIGNMENT_PLAN.md), [`project_instructions.md`](project_instructions.md).
