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

DEFAULT_IMAGE_DOWNLOAD_LIMIT = 100
DOWNLOAD_TIMEOUT_SECONDS = 10
DOWNLOAD_MAX_RETRIES = 3

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512

DEFAULT_TOP_K = 5
