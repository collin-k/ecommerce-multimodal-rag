"""
Evaluate multimodal product retrieval with Recall@k.

Text protocol: query by product name and check whether the same catalog
row is retrieved (optimistic self-retrieval).

Image protocol: query with ``product_{index}.jpg`` and check whether row
``index`` is retrieved.

Labeled NL protocol: query with hand-written questions from
``eval_queries.json`` (not exact product names) and check whether the
labeled catalog rows are retrieved.

Run from the repository root:

    python -m eval.evaluate_retrieval --sample-size 100 --top-k 10
    python -m eval.evaluate_retrieval --labeled-only --top-k 10
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

# Import CLIP-backed retriever before NumPy/FAISS-heavy use on macOS conda.
from src.retriever import ProductRetriever
from src.config import EVAL_QUERIES_PATH, EVAL_RECALL_PATH, IMAGES_DIR, PROJECT_ROOT
from src.query_rewriter import rewrite_clip_query

RECALL_CUTOFFS = (1, 5, 10)

LIMITATIONS = (
    "Limitations: text self-retrieval is an optimistic upper bound because "
    "the query is the product name itself. Image evaluation uses locally "
    "downloaded files (often the first N catalog rows), so the image set "
    "may be biased. Labeled NL recall uses hand-written questions with "
    "in-catalog gold rows; out-of-catalog refusal items are excluded "
    "because there is no relevant product to retrieve."
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


def _relative_to_project(path: Path) -> str:
    """
    Return a repo-relative path string when the file is inside the project.

    Parameters
    ----------
    path : Path
        File path to display in metrics JSON.

    Returns
    -------
    str
        Relative path, or the original path if it is outside the repo.
    """
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_eval_queries(path: Path = EVAL_QUERIES_PATH) -> List[Dict[str, object]]:
    """
    Load the hand-labeled evaluation queries.

    Parameters
    ----------
    path : Path, optional
        JSON file written for the labeled eval set.

    Returns
    -------
    list of dict
        Query records with questions and gold catalog rows.

    Raises
    ------
    FileNotFoundError
        If the labeled file is missing.
    ValueError
        If the JSON has no ``queries`` list.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Labeled eval queries not found: {path}. "
            f"Set EVAL_QUERIES_PATH or pass --eval-queries."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"{path} must contain a top-level 'queries' list.")
    return queries


def _is_labeled_text_query(query: Dict[str, object]) -> bool:
    """
    Return whether a labeled item is an in-catalog text retrieval query.

    Parameters
    ----------
    query : dict
        One record from ``eval_queries.json``.

    Returns
    -------
    bool
        True when the query is text and has at least one gold row.
    """
    if str(query.get("modality", "text")) != "text":
        return False
    relevant = query.get("relevant_indices") or []
    return len(relevant) > 0


def _query_recall_at_cutoffs(
    hit_indices: Sequence[int],
    relevant_indices: Sequence[int],
    cutoffs: Sequence[int] = RECALL_CUTOFFS,
) -> Dict[int, float]:
    """
    Compute per-query Recall@k over one or more gold catalog rows.

    Parameters
    ----------
    hit_indices : sequence of int
        Retrieved catalog rows in rank order.
    relevant_indices : sequence of int
        Gold catalog rows for this query.
    cutoffs : sequence of int, optional
        Recall cutoffs such as 1, 5, and 10.

    Returns
    -------
    dict
        Mapping from cutoff k to the fraction of gold rows in the top-k.
    """
    relevant = {int(index) for index in relevant_indices}
    n_relevant = len(relevant)
    if n_relevant == 0:
        raise ValueError("relevant_indices cannot be empty.")

    ranked = [int(index) for index in hit_indices]
    return {
        int(cutoff): float(
            len(relevant.intersection(ranked[:cutoff])) / n_relevant
        )
        for cutoff in cutoffs
    }


def _mean_recall(
    per_query_recalls: Sequence[Dict[int, float]],
) -> Dict[int, float]:
    """
    Average per-query Recall@k dictionaries.

    Parameters
    ----------
    per_query_recalls : sequence of dict
        One Recall@k mapping per query.

    Returns
    -------
    dict
        Mean Recall@k over queries.
    """
    if not per_query_recalls:
        return {int(cutoff): 0.0 for cutoff in RECALL_CUTOFFS}

    n_queries = len(per_query_recalls)
    return {
        int(cutoff): float(
            sum(item[cutoff] for item in per_query_recalls) / n_queries
        )
        for cutoff in RECALL_CUTOFFS
    }


def evaluate_labeled_text_retrieval(
    retriever: ProductRetriever,
    queries_path: Path = EVAL_QUERIES_PATH,
    top_k: int = 10,
) -> Dict[str, object]:
    """
    Measure retrieval for hand-labeled natural-language questions.

    Skips image queries and out-of-catalog items that have no gold row.
    Comparison questions with two gold products score the fraction of
    those products found in the top-k.

    Parameters
    ----------
    retriever : ProductRetriever
        Loaded multimodal retriever.
    queries_path : Path, optional
        Labeled query JSON.
    top_k : int, optional
        Retrieval depth for each query.

    Returns
    -------
    dict
        Recall@k, per-intent breakdown, and per-query ranks.
    """
    labeled = [
        query
        for query in load_eval_queries(queries_path)
        if _is_labeled_text_query(query)
    ]
    if not labeled:
        raise ValueError(f"No in-catalog text queries found in {queries_path}.")

    per_query_recalls: List[Dict[int, float]] = []
    details: List[Dict[str, object]] = []
    by_intent_recalls: Dict[str, List[Dict[int, float]]] = defaultdict(list)

    for query in labeled:
        question = str(query["question"]).strip()
        relevant = [int(index) for index in query["relevant_indices"]]
        hits = retriever.search_text(question, top_k=top_k, rewrite=True)
        hit_indices = [product.index for product in hits]
        ranks = [
            _rank_of_hit(hit_indices, relevant_index, top_k)
            for relevant_index in relevant
        ]
        query_recall = _query_recall_at_cutoffs(hit_indices, relevant)
        intent = str(query.get("intent", "unknown"))

        per_query_recalls.append(query_recall)
        by_intent_recalls[intent].append(query_recall)
        details.append(
            {
                "id": query["id"],
                "intent": intent,
                "relevant_indices": relevant,
                "clip_query": rewrite_clip_query(question),
                "ranks": ranks,
                "recall": query_recall,
            }
        )

    by_intent = {
        intent: {
            "n_queries": len(recalls),
            "recall": _mean_recall(recalls),
        }
        for intent, recalls in sorted(by_intent_recalls.items())
    }

    return {
        "protocol": "labeled_nl_text",
        "query_rewrite": True,
        "n_queries": len(labeled),
        "n_relevant_items": sum(
            len(query["relevant_indices"]) for query in labeled
        ),
        "top_k": top_k,
        "queries_path": _relative_to_project(queries_path),
        "recall": _mean_recall(per_query_recalls),
        "by_intent": by_intent,
        "queries": details,
    }


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
    if "n_relevant_items" in result:
        print(f"  gold items: {result['n_relevant_items']}")
    recall = result["recall"]
    for cutoff in RECALL_CUTOFFS:
        print(f"  Recall@{cutoff}: {recall[cutoff]:.3f}")

    by_intent = result.get("by_intent")
    if not isinstance(by_intent, dict):
        return
    for intent, payload in by_intent.items():
        intent_recall = payload["recall"]
        print(
            f"  [{intent}] n={payload['n_queries']}  "
            + "  ".join(
                f"R@{cutoff}={intent_recall[cutoff]:.3f}"
                for cutoff in RECALL_CUTOFFS
            )
        )


def _json_ready(result: Dict[str, object], *, include_queries: bool = False) -> Dict[str, object]:
    """
    Drop bulky fields before writing metrics JSON.

    Parameters
    ----------
    result : dict
        Evaluation payload.
    include_queries : bool, optional
        Keep per-query ranks when True (labeled protocol is small).

    Returns
    -------
    dict
        JSON-serializable metrics.
    """
    skip = set() if include_queries else {"queries", "ranks"}
    return {key: value for key, value in result.items() if key not in skip}


def main() -> None:
    """Run text, image, and labeled NL Recall@k evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate product retrieval.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-queries",
        type=str,
        default=str(EVAL_QUERIES_PATH),
        help="Path to labeled eval_queries.json.",
    )
    parser.add_argument(
        "--labeled-only",
        action="store_true",
        help="Run only the labeled NL protocol and merge into the output file.",
    )
    parser.add_argument(
        "--skip-labeled",
        action="store_true",
        help="Skip the labeled NL protocol.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EVAL_RECALL_PATH),
        help="JSON path for saved metrics.",
    )
    args = parser.parse_args()

    if args.top_k < max(RECALL_CUTOFFS):
        raise ValueError(
            f"--top-k must be at least {max(RECALL_CUTOFFS)} to report Recall@10."
        )

    retriever = ProductRetriever()
    output_path = Path(args.output)
    payload: Dict[str, object] = {}

    if args.labeled_only and output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))

    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["limitations"] = LIMITATIONS

    if not args.labeled_only:
        text_result = evaluate_text_self_retrieval(
            retriever,
            sample_size=args.sample_size,
            top_k=args.top_k,
            seed=args.seed,
        )
        _print_recall_table("Text self-retrieval Recall@k", text_result)
        payload["text"] = _json_ready(text_result)

        image_paths = list(IMAGES_DIR.glob("product_*.jpg"))
        if image_paths:
            image_result = evaluate_image_to_product(
                retriever,
                top_k=args.top_k,
            )
            _print_recall_table("Image retrieval Recall@k", image_result)
            payload["image"] = _json_ready(image_result)
        else:
            print("Skipping image evaluation; no local product images found.")

    if not args.skip_labeled:
        queries_path = Path(args.eval_queries)
        labeled_result = evaluate_labeled_text_retrieval(
            retriever,
            queries_path=queries_path,
            top_k=args.top_k,
        )
        _print_recall_table("Labeled NL text Recall@k", labeled_result)
        existing_labeled = payload.get("labeled_text")
        if (
            isinstance(existing_labeled, dict)
            and not existing_labeled.get("query_rewrite")
            and "labeled_text_baseline" not in payload
        ):
            payload["labeled_text_baseline"] = existing_labeled
        payload["labeled_text"] = _json_ready(
            labeled_result,
            include_queries=True,
        )

    print()
    print(LIMITATIONS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
