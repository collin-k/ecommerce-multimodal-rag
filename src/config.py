"""
Project-wide configuration for the multimodal e-commerce RAG system.

Paths resolve relative to the repository root so scripts work from any cwd.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"

PRODUCTS_CSV = PROCESSED_DIR / "clean_products.csv"
TEXT_EMBEDDINGS_PATH = PROCESSED_DIR / "text_embeddings.npy"
FAISS_INDEX_PATH = PROCESSED_DIR / "products_faiss.index"
IMAGE_EMBEDDINGS_PATH = PROCESSED_DIR / "test_image_embeddings.npy"
EVAL_QUERIES_PATH = PROCESSED_DIR / "eval_queries.json"
EVAL_RAG_PATH = PROCESSED_DIR / "eval_rag.json"

DEFAULT_IMAGE_DOWNLOAD_LIMIT = 100
DOWNLOAD_TIMEOUT_SECONDS = 10
DOWNLOAD_MAX_RETRIES = 3

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512

DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARACTERS = 6000

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2

SYSTEM_INSTRUCTIONS = (
    "You are a helpful e-commerce product assistant. "
    "Answer using only the retrieved product context. "
    "If the context is insufficient or the requested product is not present, "
    "say you do not have enough information and do not invent products, "
    "prices, or specifications. "
    "Be concise, accurate, and cite product names from the context."
)

COMPARISON_FEW_SHOT = (
    "Example comparison style:\n"
    "User: Compare Product A with Product B.\n"
    "Assistant: Product A (from the catalog) has ... Product B is not in the "
    "retrieved catalog context, so I cannot compare the two from this data.\n"
)
