"""
Score RAG answers on labeled queries.

Automatic checks cover retrieval hit, a conservative groundedness
heuristic, and out-of-catalog refusal. Each row also has empty ``hand``
fields for a human checklist (relevant / grounded / complete / refused).

Run from the repository root:

    python -m src.evaluate_rag --top-k 5
    python -m src.evaluate_rag --no-llm
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Import CLIP-backed retriever before FAISS / OpenAI client setup.
from .retriever import ProductRetriever, RetrievedProduct
from .config import (
    DEFAULT_TOP_K,
    EVAL_QUERIES_PATH,
    EVAL_RAG_PATH,
    PROJECT_ROOT,
)
from .evaluate_retrieval import load_eval_queries
from .rag import ProductRagAssistant, require_api_key

REFUSAL_PHRASES = (
    "do not have enough",
    "don't have enough",
    "does not have enough",
    "not in the retrieved",
    "not in the catalog",
    "not in this catalog",
    "not present",
    "not available",
    "cannot compare",
    "can't compare",
    "could not find",
    "couldn't find",
    "no matching",
    "insufficient",
    "do not invent",
    "i don't know",
    "i do not know",
)

SPEC_HINTS = (
    "$",
    "mah",
    "battery",
    "display",
    "bluetooth",
    "wi-fi",
    "wifi",
    "noise cancellation",
    "megapixel",
    "processor",
)

GENERIC_NAME_TOKENS = frozenset(
    {
        "with",
        "and",
        "the",
        "for",
        "from",
        "inch",
        "inches",
        "piece",
        "pieces",
        "kids",
        "child",
        "set",
        "new",
        "available",
        "complete",
        "pack",
        "over",
        "this",
        "that",
        "into",
        "your",
        "kit",
        "toys",
        "games",
        "product",
    }
)

LIMITATIONS = (
    "Automatic groundedness is a heuristic: it flags invented missing-product "
    "specs and prices not present in retrieved context. It cannot catch every "
    "subtle hallucination. Fill the per-query 'hand' fields for the graded "
    "scorecard. Out-of-catalog items have no retrieval gold."
)


def looks_like_refusal(answer: str) -> bool:
    """
    Return whether the answer uses catalog-gap / insufficient-info language.

    Parameters
    ----------
    answer : str
        Model output.

    Returns
    -------
    bool
        True when a refusal phrase is present.
    """
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def _name_cues(product_name: str) -> List[str]:
    """
    Build distinctive substrings used to detect a product mention.

    Parameters
    ----------
    product_name : str
        Catalog product title.

    Returns
    -------
    list of str
        Lowercased cues, longest first.
    """
    cleaned = product_name.lower().strip()
    tokens = [
        token.strip("\"',.-()")
        for token in cleaned.split()
        if len(token.strip("\"',.-()")) > 3
        and token.strip("\"',.-()") not in GENERIC_NAME_TOKENS
    ]
    cues: List[str] = []
    if len(cleaned) >= 16:
        cues.append(cleaned[:28].strip())
    if len(tokens) >= 2:
        cues.append(" ".join(tokens[:2]))
    if tokens:
        longest = max(tokens, key=len)
        if len(longest) >= 6:
            cues.append(longest)
    return cues


def cites_product_name(answer: str, names: Sequence[str]) -> bool:
    """
    Return whether the answer mentions any of the given product names.

    Parameters
    ----------
    answer : str
        Model output.
    names : sequence of str
        Catalog titles to look for.

    Returns
    -------
    bool
        True when a name or a distinctive cue appears.
    """
    lowered = answer.lower()
    for name in names:
        if name.lower() in lowered:
            return True
        for cue in _name_cues(name):
            if cue and cue in lowered:
                return True
    return False


def mentions_missing_product(answer: str, missing_names: Sequence[str]) -> bool:
    """
    Return whether the answer names a product that is not a gold hit.

    Parameters
    ----------
    answer : str
        Model output.
    missing_names : sequence of str
        Brands or titles that must not be treated as in-catalog.

    Returns
    -------
    bool
        True when a missing name string appears.
    """
    lowered = answer.lower()
    return any(name.lower() in lowered for name in missing_names if name.strip())


def invented_prices(answer: str, context: str) -> bool:
    """
    Return whether the answer cites a dollar amount absent from context.

    Parameters
    ----------
    answer : str
        Model output.
    context : str
        Retrieved product text sent to the LLM.

    Returns
    -------
    bool
        True when a price in the answer is not in the context.
    """
    prices = re.findall(r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?", answer)
    if not prices:
        return False
    compact_context = context.replace(" ", "")
    for price in prices:
        if price.replace(" ", "") not in compact_context and price not in context:
            return True
    return False


def invented_missing_specs(answer: str, missing_names: Sequence[str]) -> bool:
    """
    Return whether the answer asserts specs for a missing product.

    Naming the missing product while refusing is allowed. Naming it and
    adding specification language without a refusal is treated as invention.

    Parameters
    ----------
    answer : str
        Model output.
    missing_names : sequence of str
        Out-of-catalog titles.

    Returns
    -------
    bool
        True when missing-product specs appear without a refusal.
    """
    if not mentions_missing_product(answer, missing_names):
        return False
    if looks_like_refusal(answer):
        return False
    lowered = answer.lower()
    return any(hint in lowered for hint in SPEC_HINTS)


def score_retrieval(
    retrieved: Sequence[RetrievedProduct],
    relevant_indices: Sequence[int],
) -> Dict[str, Any]:
    """
    Score whether gold catalog rows appear in the retrieved list.

    Parameters
    ----------
    retrieved : sequence of RetrievedProduct
        Ranked retrieval hits.
    relevant_indices : sequence of int
        Gold catalog rows. Empty for out-of-catalog queries.

    Returns
    -------
    dict
        Hit flags, recall, and retrieved indices.
    """
    hit_indices = [product.index for product in retrieved]
    relevant = [int(index) for index in relevant_indices]
    if not relevant:
        return {
            "applicable": False,
            "hit_all": None,
            "hit_any": None,
            "recall": None,
            "retrieved_indices": hit_indices,
        }

    relevant_set = set(relevant)
    retrieved_set = set(hit_indices)
    n_found = len(relevant_set.intersection(retrieved_set))
    return {
        "applicable": True,
        "hit_all": n_found == len(relevant_set),
        "hit_any": n_found > 0,
        "recall": n_found / len(relevant_set),
        "retrieved_indices": hit_indices,
    }


def auto_grounded(
    answer: str,
    context: str,
    retrieved_names: Sequence[str],
    relevant_names: Sequence[str],
    missing_names: Sequence[str],
    must_refuse: bool,
) -> bool:
    """
    Apply the conservative automatic groundedness heuristic.

    Parameters
    ----------
    answer : str
        Model output.
    context : str
        Retrieved context given to the LLM.
    retrieved_names : sequence of str
        Titles of retrieved products.
    relevant_names : sequence of str
        Gold in-catalog titles.
    missing_names : sequence of str
        Titles that must not be invented.
    must_refuse : bool
        True when the query is fully out of catalog.

    Returns
    -------
    bool
        True when the heuristic finds no clear invention.
    """
    if not answer.strip():
        return False
    if context and invented_prices(answer, context):
        return False
    if invented_missing_specs(answer, missing_names):
        return False
    if looks_like_refusal(answer):
        return True
    if relevant_names:
        return cites_product_name(answer, relevant_names) or cites_product_name(
            answer,
            retrieved_names,
        )
    return True


def auto_ood_refused(
    answer: str,
    missing_names: Sequence[str],
    must_refuse: bool,
) -> Optional[bool]:
    """
    Score out-of-catalog refusal when the query requires it.

    Parameters
    ----------
    answer : str
        Model output.
    missing_names : sequence of str
        Titles that are not in the catalog.
    must_refuse : bool
        Whether this query is labeled as a required refusal.

    Returns
    -------
    bool or None
        True/False for refusal queries, otherwise None.
    """
    if not must_refuse:
        return None
    if not answer.strip():
        return False
    if invented_missing_specs(answer, missing_names):
        return False
    return looks_like_refusal(answer)


def _empty_hand_scores() -> Dict[str, Optional[object]]:
    """
    Return blank human-checklist fields.

    Returns
    -------
    dict
        ``relevant``, ``grounded``, ``complete``, ``refused_ood``, ``notes``.
    """
    return {
        "relevant": None,
        "grounded": None,
        "complete": None,
        "refused_ood": None,
        "notes": "",
    }


def _resolve_eval_image(image_value: object) -> Optional[Path]:
    """
    Resolve a labeled image path against the repository root.

    Parameters
    ----------
    image_value : object
        Relative path string from the labeled JSON, or None.

    Returns
    -------
    Path or None
        Existing image path, or None when not provided.
    """
    if not image_value:
        return None
    image_path = Path(str(image_value))
    if not image_path.is_file():
        image_path = PROJECT_ROOT / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Eval image not found: {image_value}")
    return image_path


def _answer_query(
    assistant: Optional[ProductRagAssistant],
    retriever: ProductRetriever,
    query: Dict[str, object],
    top_k: int,
    use_llm: bool,
) -> Dict[str, Any]:
    """
    Run retrieval, and generation when an assistant is provided.

    Parameters
    ----------
    assistant : ProductRagAssistant or None
        RAG assistant. Ignored when ``use_llm`` is False.
    retriever : ProductRetriever
        Shared retriever.
    query : dict
        Labeled eval record.
    top_k : int
        Retrieval depth.
    use_llm : bool
        When False, skip generation and leave the answer empty.

    Returns
    -------
    dict
        Answer, context, and retrieved products.
    """
    question = str(query["question"]).strip()
    image_path = _resolve_eval_image(query.get("image"))

    if use_llm:
        if assistant is None:
            raise ValueError("Assistant is required when use_llm is True.")
        if image_path is not None:
            return assistant.answer_image(question, image_path, top_k=top_k)
        return assistant.answer_text(question, top_k=top_k)

    if image_path is not None:
        products = retriever.search_image(image_path, top_k=top_k)
    else:
        products = retriever.search_text(question, top_k=top_k)
    return {
        "answer": "",
        "products": products,
        "model": None,
        "context": "",
    }


def evaluate_one_query(
    query: Dict[str, object],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach automatic scores and a blank hand checklist to one RAG result.

    Parameters
    ----------
    query : dict
        Labeled eval record.
    result : dict
        Assistant (or retrieval-only) payload.

    Returns
    -------
    dict
        Scorecard row for JSON output.
    """
    products: List[RetrievedProduct] = list(result["products"])
    answer = str(result.get("answer") or "")
    context = str(result.get("context") or "")
    relevant_indices = [int(index) for index in (query.get("relevant_indices") or [])]
    relevant_names = [str(name) for name in (query.get("relevant_names") or [])]
    missing_names = [str(name) for name in (query.get("missing_names") or [])]
    must_refuse = bool(query.get("must_refuse"))
    retrieved_names = [product.product_name for product in products]
    retrieval = score_retrieval(products, relevant_indices)
    gold_in_context = False
    if context and relevant_names:
        gold_in_context = any(
            name.lower() in context.lower() for name in relevant_names
        )

    grounded = None
    refused = None
    if answer:
        grounded = auto_grounded(
            answer,
            context,
            retrieved_names,
            relevant_names,
            missing_names,
            must_refuse,
        )
        refused = auto_ood_refused(answer, missing_names, must_refuse)

    return {
        "id": query["id"],
        "intent": query.get("intent"),
        "modality": query.get("modality", "text"),
        "question": query["question"],
        "must_refuse": must_refuse,
        "answer": answer,
        "model": result.get("model"),
        "retrieved_names": retrieved_names,
        "retrieval": retrieval,
        "gold_in_context": gold_in_context if relevant_names else None,
        "auto": {
            "cites_gold_name": cites_product_name(answer, relevant_names)
            if answer and relevant_names
            else None,
            "mentions_missing_name": mentions_missing_product(answer, missing_names)
            if answer
            else None,
            "looks_like_refusal": looks_like_refusal(answer) if answer else None,
            "invented_prices": invented_prices(answer, context) if answer else None,
            "grounded": grounded,
            "ood_refused": refused,
        },
        "hand": _empty_hand_scores(),
    }


def _mean(values: Sequence[Optional[bool]]) -> Optional[float]:
    """
    Average boolean scores, ignoring None.

    Parameters
    ----------
    values : sequence of bool or None
        Per-query flags.

    Returns
    -------
    float or None
        Mean in ``[0, 1]``, or None when no values apply.
    """
    scored = [float(value) for value in values if value is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def summarize_scorecard(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate automatic RAG metrics.

    Parameters
    ----------
    rows : sequence of dict
        Per-query scorecard rows.

    Returns
    -------
    dict
        Headline rates and counts.
    """
    retrieval_hits = [
        row["retrieval"]["hit_all"]
        for row in rows
        if row["retrieval"]["applicable"]
    ]
    return {
        "n_queries": len(rows),
        "n_with_gold": sum(row["retrieval"]["applicable"] for row in rows),
        "n_must_refuse": sum(bool(row["must_refuse"]) for row in rows),
        "n_with_answers": sum(bool(row["answer"]) for row in rows),
        "retrieval_hit": _mean(retrieval_hits),
        "gold_in_context": _mean(
            [
                row["gold_in_context"]
                for row in rows
                if row["gold_in_context"] is not None
            ]
        ),
        "grounded": _mean(row["auto"]["grounded"] for row in rows),
        "ood_refusal": _mean(row["auto"]["ood_refused"] for row in rows),
        "hand_scores_pending": True,
    }


def evaluate_rag(
    queries_path: Path = EVAL_QUERIES_PATH,
    top_k: int = DEFAULT_TOP_K,
    use_llm: bool = True,
    skip_images: bool = False,
    max_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run labeled queries through retrieval and optional RAG generation.

    Parameters
    ----------
    queries_path : Path, optional
        Labeled eval JSON.
    top_k : int, optional
        Retrieval depth (defaults to the assistant's ``DEFAULT_TOP_K``).
    use_llm : bool, optional
        When False, score retrieval only and leave answers blank.
    skip_images : bool, optional
        Drop image-modality queries.
    max_queries : int or None, optional
        Evaluate only the first N remaining queries.

    Returns
    -------
    dict
        Metrics plus per-query rows with empty hand-checklist fields.
    """
    queries = load_eval_queries(queries_path)
    if skip_images:
        queries = [
            query
            for query in queries
            if str(query.get("modality", "text")) != "image"
        ]
    if max_queries is not None:
        queries = queries[: max(0, max_queries)]

    print("Loading retriever...")
    retriever = ProductRetriever()
    assistant: Optional[ProductRagAssistant] = None
    model_name = None
    if use_llm:
        require_api_key()
        print("Loading RAG assistant...")
        assistant = ProductRagAssistant(retriever=retriever)
        model_name = assistant.model_name

    rows: List[Dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        query_id = query.get("id", index)
        print(f"[{index}/{len(queries)}] {query_id}")
        try:
            result = _answer_query(
                assistant,
                retriever,
                query,
                top_k=top_k,
                use_llm=use_llm,
            )
        except FileNotFoundError as error:
            print(f"  skipped: {error}")
            continue
        rows.append(evaluate_one_query(query, result))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "labeled_rag",
        "model": model_name,
        "top_k": top_k,
        "use_llm": use_llm,
        "queries_path": "data/processed/eval_queries.json",
        "limitations": LIMITATIONS,
        "metrics": summarize_scorecard(rows),
        "queries": rows,
    }


def _format_rate(value: Optional[float]) -> str:
    """
    Format an optional rate for the console.

    Parameters
    ----------
    value : float or None
        Mean in ``[0, 1]``.

    Returns
    -------
    str
        Three-decimal rate, or ``n/a``.
    """
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _print_scorecard(payload: Dict[str, Any]) -> None:
    """
    Print headline RAG metrics.

    Parameters
    ----------
    payload : dict
        Output of ``evaluate_rag``.
    """
    metrics = payload["metrics"]
    print("RAG scorecard (automatic)")
    print(f"  queries: {metrics['n_queries']}")
    print(f"  with answers: {metrics['n_with_answers']}")
    print(f"  retrieval hit (all gold in top-k): {_format_rate(metrics['retrieval_hit'])}")
    print(f"  gold in LLM context: {_format_rate(metrics['gold_in_context'])}")
    print(f"  grounded (heuristic): {_format_rate(metrics['grounded'])}")
    print(f"  OOD refusal: {_format_rate(metrics['ood_refusal'])}")
    print("  hand checklist: fill queries[].hand in the JSON output")


def main() -> None:
    """Run the labeled RAG scorecard and write JSON."""
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers on labeled product queries.",
    )
    parser.add_argument(
        "--eval-queries",
        type=str,
        default=str(EVAL_QUERIES_PATH),
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Score retrieval only; leave answers blank for a hand checklist.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image-modality queries.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries (smoke test).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EVAL_RAG_PATH),
    )
    args = parser.parse_args()

    payload = evaluate_rag(
        queries_path=Path(args.eval_queries),
        top_k=args.top_k,
        use_llm=not args.no_llm,
        skip_images=args.skip_images,
        max_queries=args.max_queries,
    )
    _print_scorecard(payload)
    print()
    print(LIMITATIONS)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
