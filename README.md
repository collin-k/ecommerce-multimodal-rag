# ecommerce-multimodal-rag

Multimodal conversational AI for e-commerce: CLIP + FAISS retrieval and RAG over the Amazon Product Dataset 2020. Users can ask with **text**, upload a **product image**, or both.

![Streamlit chatbot screenshot](docs/screenshot.png)

*Add a screenshot of `streamlit run src/app.py` as `docs/screenshot.png` (optional; gitignored images are fine to keep local).*

## Architecture

```text
User (text and/or image)
        │
        ▼
┌───────────────────┐
│  Streamlit app    │  src/app.py
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  ProductRag       │  src/rag.py  → OpenAI-compatible LLM
│  Assistant        │
└─────────┬─────────┘
          │ top-k products as context
          ▼
┌───────────────────┐
│ ProductRetriever  │  src/retriever.py
│  search_text()    │
│  search_image()   │──► CLIP image embed ──┐
└─────────┬─────────┘                       │
          │                                 │
          ▼                                 ▼
   CLIP text embed              shared 512-d space
          │                                 │
          └────────────┬────────────────────┘
                       ▼
              FAISS IndexFlatIP
           (text-product vectors)
                       │
                       ▼
            clean_products.csv rows
```

CLIP maps images and text into one space, so an image query can search the **text-product** FAISS index directly.

## Setup

```bash
conda activate base   # or any env with the packages in requirements.txt

cd ecommerce-multimodal-rag
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY=...
# optional: OPENAI_BASE_URL and LLM_MODEL for Groq/Together Llama 3.1
```

Python env used in development: conda `base` with `torch`, `transformers`, `faiss-cpu`, `streamlit`, and `openai`.

## Data artifacts

| Path | In git? | Notes |
|------|---------|--------|
| `data/processed/clean_products.csv` | **Yes** | Clean ~10k-product catalog |
| `data/raw/` | No | Original Kaggle dump |
| `data/processed/*.npy` | No | CLIP embeddings |
| `data/processed/*.index` | No | FAISS index |
| `data/images/` | No | Product JPEGs |
| `.env` | No | Secrets — copy from `.env.example` |

If embeddings or the FAISS index are missing locally, rebuild them from the clean CSV:

```bash
python src/clip_embeddings.py
python src/build_faiss.py
python -m src.image_downloader --limit 100
```

Dataset: [Amazon Product Dataset 2020](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020).

## Run the chatbot

From the repository root (`src/app.py` is on the Streamlit branch if it is not already on `main`):

```bash
streamlit run src/app.py
```

Modes:

- **Text question** — natural-language Q&A over retrieved catalog rows (needs `.env`)
- **Image question** — upload a product photo to identify / explain it (needs `.env`)
- **Retrieve only** — CLIP + FAISS search without calling the LLM

## Other commands

```bash
python -m src.search_products --query "500 piece jigsaw puzzle"
python -m src.search_products --image data/images/product_0.jpg
python -m src.rag --question "What are the features of the DB Longboards CoreFlex Crossbow?"
python -m src.rag --question "What is this?" --image data/images/product_0.jpg
python -m src.evaluate_retrieval --sample-size 100 --top-k 10
```

## Known limitations

- The catalog is a 10k Amazon 2020 sample. Famous demo products from the assignment (Galaxy S21, AirPods Pro, Echo Dot) are often **absent**; the LLM is instructed not to invent them.
- Image queries search the **text** FAISS index via CLIP. That is enough for identification demos, but Recall@1 is much lower than text self-retrieval.
- Text Recall@k uses product-name self-retrieval (optimistic). Image eval uses the first N downloaded rows.
- First CLIP load is slow. On some macOS/conda stacks, FAISS must be imported after CLIP to avoid OpenMP segfaults.
- `clip_embeddings.py` / `build_faiss.py` still use cwd-relative paths; run them from the repo root.

Full writeup: [`docs/research_report.md`](docs/research_report.md). Assignment plan: [`docs/ASSIGNMENT_PLAN.md`](docs/ASSIGNMENT_PLAN.md). Course brief: [`project_instructions.md`](project_instructions.md).
