# Multimodal Conversational AI for E-commerce

**Course:** GEN AI PRINCIPLES — Project II  
**System:** CLIP + FAISS retrieval-augmented generation over the Amazon Product Dataset 2020

## 1. Introduction

E-commerce support is inherently multimodal. Shoppers ask about features and compatibility in language, but they also send photos of products they already have or want to identify. Text-only chatbots cannot ground those visual queries, so answers are incomplete and human agents absorb the leftover work.

This project builds a **vision-language RAG assistant**: CLIP encodes text and images into one vector space, FAISS retrieves nearby catalog products, and an LLM answers only from that retrieved context. The goal is accurate, catalog-grounded responses for text questions, image uploads, and “show me this product” requests.

## 2. Data

**Source.** [Amazon Product Dataset 2020](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020) (PromptCloud / Kaggle). The working catalog is a cleaned sample of **10,002** products in `data/processed/clean_products.csv`.

**Columns kept.** `Uniq Id`, `Product Name`, `Image` (pipe-separated URLs), `Product Url`, plus two derived text fields:

| Field | Role |
|-------|------|
| `combined_text` | Rich document for the LLM: name, category, price, about-product, specifications, technical details |
| `clip_text` | Shorter string used to build CLIP **text** embeddings (name, category, and truncated product copy) |

**Cleaning.** Rows were normalized into those document fields, empty names were left identifiable (some titles are `"-"`), and the first URL in `Image` is treated as the primary photo. About 100 product images were downloaded for demos and image Recall@k (`data/images/product_{row}.jpg`). Failed URLs are skipped rather than aborting the batch.

**Attribute choice.** Title, category, price, and feature-like “About Product” text were prioritized so retrieval has distinctive language and the LLM has enough specification detail to answer “what are the features of X?” without leaving the catalog.

## 3. Methods

### 3.1 CLIP encoder

`openai/clip-vit-base-patch32` produces 512-dimensional embeddings for both text and images (Radford et al., 2021). Vectors are L2-normalized so inner product equals cosine similarity. Newer `transformers` releases return `BaseModelOutputWithPooling` from `get_text_features` / `get_image_features`; the implementation reads `pooler_output` when a tensor is not returned.

### 3.2 Vector index

Catalog **text** embeddings are stored in a local FAISS `IndexFlatIP` (`data/processed/products_faiss.index`). Vertex AI Vector Search was listed as an example in the brief; a local index was sufficient for 10k vectors and keeps the project reproducible without cloud credentials.

**Image → product.** An uploaded image is encoded with CLIP and searched against the **text** index. Shared embedding space makes a second 10k image index unnecessary for identification.

### 3.3 RAG and prompting

`ProductRagAssistant` retrieves top-k products, truncates context to 6,000 characters, and calls an OpenAI-compatible chat API (`gpt-4o-mini` by default, temperature 0.2).

- **Query rewrite (CLIP only):** shopper questions are simplified to a product-centric string before embedding (`src/query_rewriter.py`). The original question is unchanged for the LLM.
- **Zero-shot system prompt:** answer only from retrieved context; admit gaps; do not invent products or prices.
- **Few-shot (comparisons):** if the question contains “compare” / “versus”, a short comparison template is appended so the model cites what is in-catalog and refuses what is not.
- **Open-source LLM path:** set `OPENAI_BASE_URL` and `LLM_MODEL` to a Groq, Together, or Ollama endpoint (e.g. Llama 3.1) to match the course’s open-source example without changing code.

### 3.4 Interface

Streamlit (`src/app.py`) exposes three modes: text question, image question, and retrieve-only (no LLM). Retrieved hits show name, score, snippet, product URL, and a local or remote image when available.

## 4. System design

```text
User (text and/or image)
        │
        ▼
  Streamlit (src/app.py)
        │
        ▼
  ProductRagAssistant (src/rag.py)
        │  top-k combined_text
        ▼
  ProductRetriever (src/retriever.py)
        │
        ├── search_text  → CLIP text embed
        └── search_image → CLIP image embed
                │
                ▼
         FAISS IndexFlatIP  (text-product vectors)
                │
                ▼
         clean_products.csv
```

| Module | Responsibility |
|--------|----------------|
| `src/config.py` | Paths, CLIP/LLM constants, prompts |
| `src/clip_encoder.py` | Shared CLIP encode for text and images |
| `src/retriever.py` | Load catalog + FAISS; `search_text` / `search_image` |
| `src/image_downloader.py` | Robust JPEG download with retries |
| `src/query_rewriter.py` | CLIP query rewrite (product name / attributes) |
| `src/rag.py` | Grounded generation + CLI |
| `src/app.py` | Streamlit chatbot |
| `eval/evaluate_retrieval.py` | Recall@1/5/10 (self, image, labeled NL) |
| `eval/evaluate_rag.py` | RAG scorecard: retrieval hit, grounded, OOD refusal |

On some macOS/conda OpenMP stacks, importing FAISS before loading CLIP segfaults. The retriever constructs CLIP first, then imports FAISS. Evaluation searches **one query vector at a time** for the same reason (batch `index.search` was unstable).

## 5. Experiments

### 5.1 Retrieval metrics

Protocol (seed 42, top-k = 10 unless noted):

- **Text self-retrieval:** sample 100 products; query = `Product Name`; relevant item = same row. Optimistic upper bound.
- **Image→product:** 100 local files `product_{i}.jpg`; relevant item = row `i`. CLIP image query vs text index.
- **Labeled NL text:** 25 hand-written questions in `eval/eval_queries.json` (not exact product names). Out-of-catalog items are excluded because there is no gold row. Comparison questions with two gold products score the fraction of those products in the top-k.

| Protocol | Queries | Recall@1 | Recall@5 | Recall@10 |
|----------|---------|----------|----------|-----------|
| Text self-retrieval | 100 | 0.930 | 0.990 | 0.990 |
| Image→product | 100 | 0.330 | 0.570 | 0.640 |
| Labeled NL (raw question) | 25 | 0.620 | 1.000 | 1.000 |
| Labeled NL (CLIP query rewrite) | 25 | **0.820** | 1.000 | 1.000 |

Text names almost always retrieve the same row in the top 5. Embedding the full question (“What are the features of …”) drops rank-1 from 0.93 to 0.62. Stripping the question template before CLIP recovers rank-1 to **0.82** without changing Recall@5/10 (already 1.0 on this set). Image identification is weaker, as expected when matching a photo to a **text** embedding of a noisy Amazon title/description. Recall@10 of 0.64 is still usable for RAG: the LLM sees several visually related neighbors.

By intent after rewrite: features R@1 0.812 (n=16), show-image R@1 1.000 (n=4), compare R@1 0.700 (n=5; two gold products cannot both be rank-1).

Reproduce:

```bash
python -m eval.evaluate_retrieval --sample-size 100 --top-k 10
python -m eval.evaluate_retrieval --labeled-only --top-k 10
```

### 5.2 RAG scorecard

`eval/evaluate_rag.py` runs the assistant on all 41 labeled queries (top-k = 5, matching the app) and scores retrieval hit, a conservative groundedness heuristic, and out-of-catalog refusal. Per-query `hand` fields are left blank for a human checklist.

| Metric | Score | Notes |
|--------|------:|-------|
| Retrieval hit (all gold in top-k) | 0.848 | 33 queries with gold rows; misses are mostly image identify/usage |
| Gold text in LLM context | 0.818 | Truncation or a missed retrieve |
| Grounded (heuristic) | 1.000 | No invented OOD specs or off-context prices |
| OOD refusal | 1.000 | All 8 must-refuse items (Galaxy S21, Echo Dot, KitchenAid, Fitbit, AirPods, Nest Mini) |

The LLM follows the grounding prompt well. Remaining quality risk is **image retrieval**, not hallucination of assignment brands. Fill `queries[].hand` in `eval/eval_rag.json` for a graded relevant/complete score.

```bash
python -m eval.evaluate_rag --top-k 5
```

### 5.3 Example interactions

These follow the assignment’s interaction types, using products that **exist** in this catalog.

**Text Q&A.**  
Question: “What are the features of the DB Longboards CoreFlex Crossbow?”  
Retrieval: rank-1 hit is `DB Longboards CoreFlex Crossbow 41" Bamboo Fiberglass Longboard Complete`.  
Expected answer: grounded in `combined_text` (category: longboards; selling price and about-product copy when present).

**Text comparison.**  
Question: “Can you compare the Amazon Echo Dot with the Google Nest Mini?”  
Retrieval: neighbors are unrelated toys/games more often than smart speakers.  
Expected answer: the assistant should say those devices are **not in the retrieved catalog context** rather than quoting the assignment’s canned Echo/Nest specs.

**Image identify / usage.**  
Input: `data/images/product_0.jpg` (longboard in the first catalog row) plus “Identify this product and describe its usage.”  
Retrieval: skateboard/longboard catalog rows (e.g. MightySkins electric-skateboard skins, Moose bamboo longboard).  
Expected answer: names a close longboard/skateboard match and usage from that row’s text—not a KitchenAid mixer from the vision PDF.

**Show me a picture.**  
Text retrieve-only (or RAG) for a name in-catalog, then the UI renders `image_url` or `data/images/product_{index}.jpg` when downloaded.

## 6. Challenges

- **Broken or blocked image URLs.** Amazon CDN links can 403 or time out. The downloader retries with a browser User-Agent and continues on failure.
- **Catalog vs. assignment examples.** Galaxy S21, AirPods Pro, Echo Dot, Nest Mini, KitchenAid, and Fitbit Charge 4 are mostly missing. Grounding rules matter more than matching the PDF’s sample answers.
- **Noisy product text.** Some names are `"-"`, CLIP strings are truncated, and categories are deep Amazon taxonomies. That hurts image-to-text matching more than name self-retrieval.
- **Latency.** CLIP load dominates the first query. IndexFlatIP over 10k × 512 is fast after that.
- **Native crashes.** FAISS + PyTorch OpenMP on conda/macOS required import-order and single-vector search workarounds.
- **CLIP API drift.** `transformers` 5.x changed `get_text_features` to return a pooling object instead of a raw tensor.

## 7. Future work

- **Vertex AI Vector Search** (or similar) if the catalog grows beyond a laptop FAISS index.
- **Full image index:** embed every product photo and support image–image search in addition to image→text.
- **Multimodal fusion** of text + image embeddings when both are present, plus a retrieval score threshold.
- **Human-graded RAG checklist** in `eval_rag.json` (`hand` fields) and optional LLM-as-judge.
- **Multi-turn memory** in Streamlit so follow-ups (“which of those is cheapest?”) reuse prior hits.
- **Rebuild scripts** (`clip_embeddings.py`, `build_faiss.py`) should use `config.py` paths and the shared `ClipEncoder`.
- **Download more than 100 images** for a less biased image Recall@k sample.

## 8. References

1. Radford, Alec, et al. “Learning Transferable Visual Models From Natural Language Supervision.” *ICML*, 2021. https://arxiv.org/abs/2103.00020  
   CLIP (Contrastive Language–Image Pre-training).

2. Liu, Z., Wang, Y., et al. “Vision-Language Alignment and Variance Adjustment.” *ICCV*, 2023. https://arxiv.org/abs/2301.00012  
   VLAVA (cited in the course brief; not used in this implementation).

3. Lewis, Patrick, et al. “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.” *NeurIPS*, 2020. https://arxiv.org/abs/2005.11401  
   RAG pattern.

4. Li, Xiangtai, et al. “Pre-trained Vision and Language Transformer for Multimodal Understanding and Generation.” *IEEE TPAMI*, 2022. https://arxiv.org/abs/2206.00020  
   Vision-language foundation models (course brief).
