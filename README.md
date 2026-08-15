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

## Run the chatbot

From the repository root:

```bash
streamlit run src/app.py
```

Modes:

- **Text question** — natural-language Q&A over retrieved catalog rows (needs `.env`)
- **Image question** — upload a product photo to identify / explain it (needs `.env`)
- **Retrieve only** — CLIP + FAISS search without calling the LLM

## Other scripts

Run from the repo root:

- `python -m src.search_products --query "500 piece jigsaw puzzle"`
- `python -m src.search_products --image data/images/product_0.jpg`
- `python -m src.image_downloader --limit 100`
- `python -m src.rag --question "What are the features of the DB Longboards CoreFlex Crossbow?"`
- `src/clip_embeddings.py` / `src/build_faiss.py` — rebuild embeddings and the FAISS index

Project scope and branch plan: [`docs/ASSIGNMENT_PLAN.md`](docs/ASSIGNMENT_PLAN.md), [`project_instructions.md`](project_instructions.md).
