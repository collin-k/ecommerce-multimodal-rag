"""
Retrieval-augmented generation over multimodal product search results.

Uses an OpenAI-compatible chat API. Point ``OPENAI_BASE_URL`` at Groq,
Together, or similar to use an open-source model such as Llama 3.1.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

try:
    from .config import (
        DEFAULT_TOP_K,
        IMAGES_DIR,
        LLM_MODEL,
        LLM_TEMPERATURE,
        MAX_CONTEXT_CHARACTERS,
        PROJECT_ROOT,
        SYSTEM_INSTRUCTIONS,
        COMPARISON_FEW_SHOT,
        IMAGE_IDENTIFY_INSTRUCTIONS,
        SHOW_IMAGE_INSTRUCTIONS,
    )
    from .retriever import ProductRetriever, RetrievedProduct, strip_amazon_boilerplate
    from .query_rewriter import rewrite_clip_query
except ImportError:
    from config import (
        DEFAULT_TOP_K,
        IMAGES_DIR,
        LLM_MODEL,
        LLM_TEMPERATURE,
        MAX_CONTEXT_CHARACTERS,
        PROJECT_ROOT,
        SYSTEM_INSTRUCTIONS,
        COMPARISON_FEW_SHOT,
        IMAGE_IDENTIFY_INSTRUCTIONS,
        SHOW_IMAGE_INSTRUCTIONS,
    )
    from retriever import ProductRetriever, RetrievedProduct, strip_amazon_boilerplate
    from query_rewriter import rewrite_clip_query

load_dotenv(PROJECT_ROOT / ".env")


def require_api_key() -> str:
    """
    Return the configured API key or raise a setup error.

    Returns
    -------
    str
        Non-empty OpenAI-compatible API key.

    Raises
    ------
    ValueError
        If ``OPENAI_API_KEY`` is missing or still the example placeholder.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY was not found.\n\n"
            f"Copy {PROJECT_ROOT / '.env.example'} to "
            f"{PROJECT_ROOT / '.env'} and set OPENAI_API_KEY."
        )
    return api_key


def format_product_context(products: List[RetrievedProduct]) -> str:
    """
    Format retrieved products into LLM context.

    Parameters
    ----------
    products : list of RetrievedProduct
        Ranked retrieval hits.

    Returns
    -------
    str
        Truncated context block for the prompt.
    """
    sections: List[str] = []

    for product in products:
        sections.append(
            f"[Product {product.rank} | score={product.score:.3f}]\n"
            f"Name: {product.product_name}\n"
            f"URL: {product.product_url}\n"
            f"Details:\n{strip_amazon_boilerplate(product.combined_text)}"
        )

    return "\n\n".join(sections)[:MAX_CONTEXT_CHARACTERS]


def is_show_image_question(question: str) -> bool:
    """
    Return whether the user asked to see a product picture.

    Parameters
    ----------
    question : str
        User message.

    Returns
    -------
    bool
        True for show-me-a-picture style requests.
    """
    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in (
            "show me a picture",
            "show me the",
            "picture of",
            "photo of",
            "image of",
        )
    )


def is_comparison_question(question: str) -> bool:
    """
    Detect comparison-style questions that benefit from few-shot guidance.

    Parameters
    ----------
    question : str
        User question text.

    Returns
    -------
    bool
        True when the question looks like a product comparison.
    """
    lowered = question.lower()
    return "compare" in lowered or " vs " in lowered or "versus" in lowered


def build_user_prompt(question: str, context: str) -> str:
    """
    Build the user message from retrieved context and the question.

    Parameters
    ----------
    question : str
        User question.
    context : str
        Formatted product context.

    Returns
    -------
    str
        Prompt sent as the user role.
    """
    parts = [f"Retrieved product context:\n{context}"]
    if is_comparison_question(question):
        parts.append(COMPARISON_FEW_SHOT)
    if is_show_image_question(question):
        parts.append(SHOW_IMAGE_INSTRUCTIONS)
    parts.append(f"User question:\n{question.strip()}")
    return "\n\n".join(parts)


class ProductRagAssistant:
    """Answer product questions using CLIP retrieval plus an LLM."""

    def __init__(
        self,
        retriever: Optional[ProductRetriever] = None,
        model_name: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        client: Optional[OpenAI] = None,
    ) -> None:
        """
        Create a multimodal product RAG assistant.

        Parameters
        ----------
        retriever : ProductRetriever or None, optional
            Shared product retriever. Created on demand when omitted.
        model_name : str or None, optional
            Chat model id. Defaults to ``LLM_MODEL`` or ``LLM_MODEL`` env.
        temperature : float, optional
            Sampling temperature for generation.
        client : OpenAI or None, optional
            Preconfigured OpenAI-compatible client.
        """
        if client is None:
            api_key = require_api_key()
            base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        else:
            self.client = client

        self.model_name = (
            model_name
            or os.getenv("LLM_MODEL")
            or LLM_MODEL
        )
        self.temperature = temperature
        self.retriever = retriever or ProductRetriever()

    def answer_text(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Answer a text product question with retrieved catalog context.

        Parameters
        ----------
        question : str
            User question about products.
        top_k : int, optional
            Number of products to retrieve.

        Returns
        -------
        dict
            Answer text, retrieved products, and prompt metadata.
        """
        products = self.retriever.search_text(
            question,
            top_k=top_k,
            rewrite=True,
        )
        result = self._generate(question, products)
        result["clip_query"] = rewrite_clip_query(question.strip())
        return result

    def answer_image(
        self,
        question: str,
        image: Union[str, Path, Image.Image],
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Answer a question about an uploaded product image.

        Parameters
        ----------
        question : str
            User question about the image.
        image : str, Path, or PIL.Image.Image
            Product image to identify and explain.
        top_k : int, optional
            Number of products to retrieve.

        Returns
        -------
        dict
            Answer text, retrieved products, and prompt metadata.
        """
        products = self.retriever.search_image(image, top_k=top_k)
        grounded_question = (
            f"{question.strip()}\n\n{IMAGE_IDENTIFY_INSTRUCTIONS}"
        )
        return self._generate(grounded_question, products)

    def _generate(
        self,
        question: str,
        products: List[RetrievedProduct],
    ) -> Dict[str, Any]:
        """
        Generate an answer from question text and retrieved products.

        Parameters
        ----------
        question : str
            User question (possibly image-augmented).
        products : list of RetrievedProduct
            Retrieval context.

        Returns
        -------
        dict
            Structured response payload.
        """
        context = format_product_context(products)
        user_prompt = build_user_prompt(question, context)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = completion.choices[0].message.content or ""

        return {
            "answer": answer.strip(),
            "products": products,
            "model": self.model_name,
            "context": context,
        }


def _resolve_image_path(image_arg: str) -> Path:
    """
    Resolve a CLI image path, including files under ``data/images``.

    Parameters
    ----------
    image_arg : str
        Path from ``--image``.

    Returns
    -------
    Path
        Existing image path.

    Raises
    ------
    FileNotFoundError
        If the image cannot be found.
    """
    image_path = Path(image_arg)
    if not image_path.is_file():
        fallback = IMAGES_DIR / image_arg
        image_path = fallback if fallback.is_file() else image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_arg}")
    return image_path


def main() -> None:
    """Run a one-off text or image RAG question from the command line."""
    parser = argparse.ArgumentParser(
        description="Answer product questions with CLIP retrieval plus an LLM.",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Optional path to a query image",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    require_api_key()
    assistant = ProductRagAssistant()

    if args.image:
        result = assistant.answer_image(
            args.question,
            _resolve_image_path(args.image),
            top_k=args.top_k,
        )
    else:
        result = assistant.answer_text(args.question, top_k=args.top_k)

    print(f"Model: {result['model']}")
    print()
    print(result["answer"])
    print()
    print("Retrieved products:")
    for product in result["products"]:
        print(f"  {product.rank}. {product.product_name} ({product.score:.3f})")


if __name__ == "__main__":
    main()
