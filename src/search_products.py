"""
CLI for text or image product retrieval against the FAISS catalog index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_TOP_K, IMAGES_DIR
from .query_rewriter import rewrite_clip_query
from .retriever import ProductRetriever


def main() -> None:
    """Run a one-off text or image retrieval query."""
    parser = argparse.ArgumentParser(description="Search the product catalog.")
    parser.add_argument("--query", type=str, default="", help="Text query")
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to a query image",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    retriever = ProductRetriever()

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            fallback = IMAGES_DIR / args.image
            image_path = fallback if fallback.is_file() else image_path

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {args.image}")

        results = retriever.search_image(image_path, top_k=args.top_k)
        print(f"Image query: {image_path}")
    elif args.query.strip():
        results = retriever.search_text(args.query, top_k=args.top_k)
        print(f"Text query: {args.query}")
        rewritten = rewrite_clip_query(args.query)
        if rewritten != args.query.strip():
            print(f"CLIP query: {rewritten}")
    else:
        default_query = "500 piece jigsaw puzzle"
        results = retriever.search_text(default_query, top_k=args.top_k)
        print(f"Text query: {default_query}")

    print()
    for product in results:
        print(f"Result {product.rank} (score={product.score:.3f}):")
        print(product.product_name)
        print(product.combined_text[:500])
        print("-" * 80)


if __name__ == "__main__":
    main()
