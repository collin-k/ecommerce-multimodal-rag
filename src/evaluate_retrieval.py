"""
Evaluate multimodal product retrieval with Recall@k.

Text protocol: query by product name and check whether the same catalog
row is retrieved (optimistic self-retrieval).

Image protocol: query with ``product_{index}.jpg`` and check whether row
``index`` is retrieved.

Run from the repository root:

    python -m src.evaluate_retrieval --sample-size 100 --top-k 10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

# Import CLIP-backed retriever before NumPy/FAISS-heavy use on macOS conda.
from .retriever import ProductRetriever
from .config import IMAGES_DIR, PROCESSED_DIR

EVAL_RESULTS_PATH = PROCESSED_DIR / "eval_recall.json"
RECALL_CUTOFFS = (1, 5, 10)

LIMITATIONS = (
    "Limitations: text self-retrieval is an optimistic upper bound because "
    "the query is the product name itself. Image evaluation uses locally "
    "downloaded files (often the first N catalog rows), so the image set "
    "may be biased. Neither protocol uses hand-labeled natural-language "
    "queries."
)


def recall_at_k(
    relevant_ranks: Sequence[int],
    cutoffs: Sequence[int] = RECALL_CUTOFFS,
) -> Dict[int, float]:
    """
    Compute Recall@k from ranks of the relevant item.

    Parameters
    ----------
    relevant_ranks : sequence of int
        1-based rank of the correct product for each query. Use a value
        larger than the largest cutoff when the item is missing.
    cutoffs : sequence of int, optional
        Recall cutoffs such as 1, 5, and 10.

    Returns
    -------
    dict
        Mapping from cutoff k to recall in ``[0, 1]``.
    """
    import numpy as np

    ranks = np.asarray(relevant_ranks, dtype=float)
    return {int(k): float(np.mean(ranks <= k)) for k in cutoffs}


def _rank_of_hit(
    hit_indices: Sequence[int],
    relevant_index: int,
    top_k: int,
) -> int:
    """
    Return the 1-based rank of the relevant row, or ``top_k + 1`` if absent.

    Parameters
    ----------
    hit_indices : sequence of int
        Retrieved catalog row indices in rank order.
    relevant_index : int
        Ground-truth catalog row.
    top_k : int
        Retrieval depth.

    Returns
    -------
    int
        Rank of the relevant item.
    """
    hit_list = [int(index) for index in hit_indices]
    if relevant_index in hit_list:
        return hit_list.index(relevant_index) + 1
    return top_k + 1


def evaluate_text_self_retrieval(
    retriever: ProductRetriever,
    sample_size: int = 100,
    top_k: int = 10,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Measure whether querying by product name retrieves the same product.

    Parameters
    ----------
    retriever : ProductRetriever
        Loaded multimodal retriever.
    sample_size : int, optional
        Number of random products to evaluate.
    top_k : int, optional
        Retrieval depth for each query.
    seed : int, optional
        Random seed for sampling.

    Returns
    -------
    dict
        Recall@k metrics, sample size, and per-query ranks.
    """
    import numpy as np

    names = retriever.products["Product Name"].fillna("").astype(str)
    valid_indices = [
        int(index)
        for index, name in enumerate(names)
        if name.strip() and name.strip() != "-"
    ]
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, len(valid_indices))
    sample_indices = rng.choice(valid_indices, size=sample_size, replace=False)
    queries = [str(names.iloc[int(index)]) for index in sample_indices]

    embeddings = retriever.encoder.encode_texts(queries)
    # Search one vector at a time. Batch FAISS search can segfault with this
    # CLIP + faiss-cpu stack on macOS.
    ranks = []
    for row, embedding in enumerate(embeddings):
        query_vector = embedding.reshape(1, -1).astype("float32")
        _, retrieved = retriever.index.search(query_vector, top_k)
        ranks.append(_rank_of_hit(retrieved[0], int(sample_indices[row]), top_k))

    metrics = recall_at_k(ranks, cutoffs=RECALL_CUTOFFS)
    return {
        "protocol": "text_self_retrieval",
        "n_queries": sample_size,
        "top_k": top_k,
        "seed": seed,
        "recall": metrics,
        "ranks": ranks,
    }


def evaluate_image_to_product(
    retriever: ProductRetriever,
    images_dir: Path = IMAGES_DIR,
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Measure whether a product image retrieves its matching catalog row.

    Expects files named ``product_{index}.jpg`` under ``images_dir``.

    Parameters
    ----------
    retriever : ProductRetriever
        Loaded multimodal retriever.
    images_dir : Path, optional
        Directory of downloaded product images.
    top_k : int, optional
        Retrieval depth for each image query.

    Returns
    -------
    dict
        Recall@k metrics, query count, and per-query ranks.
    """
    from PIL import Image

    image_paths = sorted(images_dir.glob("product_*.jpg"))
    if not image_paths:
        raise FileNotFoundError(
            f"No product_*.jpg files found in {images_dir}. "
            "Run: python -m src.image_downloader --limit 100"
        )

    row_indices: List[int] = []
    images: List[Image.Image] = []
    for image_path in image_paths:
        row_index = int(image_path.stem.split("_")[1])
        try:
            images.append(Image.open(image_path).convert("RGB"))
            row_indices.append(row_index)
        except Exception as error:
            print(f"Skipped {image_path.name}: {error}")

    embeddings = retriever.encoder.encode_images(images)
    ranks = []
    for row, embedding in enumerate(embeddings):
        query_vector = embedding.reshape(1, -1).astype("float32")
        _, retrieved = retriever.index.search(query_vector, top_k)
        ranks.append(_rank_of_hit(retrieved[0], row_indices[row], top_k))
    metrics = recall_at_k(ranks, cutoffs=RECALL_CUTOFFS)
    return {
        "protocol": "image_to_product",
        "n_queries": len(row_indices),
        "top_k": top_k,
        "recall": metrics,
        "ranks": ranks,
    }


def _print_recall_table(title: str, result: Dict[str, object]) -> None:
    """
    Print Recall@k values for one evaluation protocol.

    Parameters
    ----------
    title : str
        Section heading.
    result : dict
        Evaluation payload with ``recall`` and ``n_queries``.
    """
    print(title)
    print(f"  queries: {result['n_queries']}")
    recall = result["recall"]
    for cutoff in RECALL_CUTOFFS:
        print(f"  Recall@{cutoff}: {recall[cutoff]:.3f}")


def main() -> None:
    """Run text and image Recall@k evaluation and print results."""
    parser = argparse.ArgumentParser(description="Evaluate product retrieval.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=str(EVAL_RESULTS_PATH),
        help="JSON path for saved metrics.",
    )
    args = parser.parse_args()

    if args.top_k < max(RECALL_CUTOFFS):
        raise ValueError(
            f"--top-k must be at least {max(RECALL_CUTOFFS)} to report Recall@10."
        )

    print("Loading retriever...")
    retriever = ProductRetriever()

    print(f"Evaluating text self-retrieval on {args.sample_size} products...")
    text_result = evaluate_text_self_retrieval(
        retriever,
        sample_size=args.sample_size,
        top_k=args.top_k,
        seed=args.seed,
    )
    _print_recall_table("Text retrieval Recall@k", text_result)

    payload: Dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limitations": LIMITATIONS,
        "text": {
            key: value
            for key, value in text_result.items()
            if key != "ranks"
        },
    }

    image_paths = list(IMAGES_DIR.glob("product_*.jpg"))
    if image_paths:
        print(
            f"Evaluating image-to-product retrieval on {len(image_paths)} images..."
        )
        image_result = evaluate_image_to_product(
            retriever,
            top_k=args.top_k,
        )
        _print_recall_table("Image retrieval Recall@k", image_result)
        payload["image"] = {
            key: value
            for key, value in image_result.items()
            if key != "ranks"
        }
    else:
        print("Skipping image evaluation; no local product images found.")

    print()
    print(LIMITATIONS)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
