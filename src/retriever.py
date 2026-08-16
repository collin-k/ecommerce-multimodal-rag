"""
Product catalog loading and multimodal FAISS retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

_AMAZON_FIT_DISCLAIMER = re.compile(
    r"Make sure this fits by entering your model number\.?\s*",
    re.IGNORECASE,
)
_ABOUT_LEADING_PIPE = re.compile(r"(About Product:\s*)\|+\s*")


def strip_amazon_boilerplate(text: str) -> str:
    """
    Remove Amazon's model-number fit disclaimer from catalog copy.

    Parameters
    ----------
    text : str
        Raw ``combined_text`` or an about-product snippet.

    Returns
    -------
    str
        Text with the disclaimer and leftover leading pipes removed.
    """
    cleaned = _AMAZON_FIT_DISCLAIMER.sub("", text)
    cleaned = _ABOUT_LEADING_PIPE.sub(r"\1", cleaned)
    cleaned = re.sub(r"^\s*\|\s*", "", cleaned.strip())
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()

# Import torch-backed CLIP before FAISS. Loading FAISS first can segfault on some
# macOS / conda OpenMP stacks when CLIP is initialized afterward.
try:
    from .clip_encoder import ClipEncoder
    from .config import (
        DEFAULT_TOP_K,
        FAISS_INDEX_PATH,
        PRODUCTS_CSV,
    )
    from .query_rewriter import rewrite_clip_query
except ImportError:
    from clip_encoder import ClipEncoder
    from config import (
        DEFAULT_TOP_K,
        FAISS_INDEX_PATH,
        PRODUCTS_CSV,
    )
    from query_rewriter import rewrite_clip_query


@dataclass(frozen=True)
class RetrievedProduct:
    """One ranked product from the retriever."""

    rank: int
    index: int
    score: float
    uniq_id: str
    product_name: str
    combined_text: str
    image_url: str
    product_url: str


class ProductRetriever:
    """Retrieve products with CLIP text or image queries against a FAISS index."""

    def __init__(
        self,
        products_csv: Path = PRODUCTS_CSV,
        faiss_index_path: Path = FAISS_INDEX_PATH,
        encoder: Optional[ClipEncoder] = None,
    ) -> None:
        """
        Load the product table, FAISS index, and CLIP encoder.

        Parameters
        ----------
        products_csv : Path, optional
            Cleaned product catalog CSV.
        faiss_index_path : Path, optional
            FAISS index built from CLIP text embeddings.
        encoder : ClipEncoder or None, optional
            Shared CLIP encoder. Created on demand when omitted.
        """
        # Load CLIP before importing FAISS. Reversing that order can segfault
        # on some macOS / conda OpenMP stacks.
        self.encoder = encoder or ClipEncoder()
        import faiss

        self.products = pd.read_csv(products_csv)
        self.index = faiss.read_index(str(faiss_index_path))

        if len(self.products) != self.index.ntotal:
            raise ValueError(
                f"Product count ({len(self.products)}) does not match "
                f"FAISS vectors ({self.index.ntotal})."
            )

    def search_text(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rewrite: bool = True,
    ) -> List[RetrievedProduct]:
        """
        Retrieve products for a natural-language query.

        Parameters
        ----------
        query : str
            User text question or product description.
        top_k : int, optional
            Number of products to return.
        rewrite : bool, optional
            When True, simplify the question into a short CLIP query
            (product name / attributes) before embedding.

        Returns
        -------
        list of RetrievedProduct
            Ranked matches from the catalog.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty.")

        clip_query = rewrite_clip_query(clean_query) if rewrite else clean_query
        embedding = self.encoder.encode_texts([clip_query])
        return self._search_embedding(embedding, top_k)

    def search_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[RetrievedProduct]:
        """
        Retrieve products for an uploaded product image.

        CLIP maps images and text into one space, so an image query can search
        the text-product FAISS index directly.

        Parameters
        ----------
        image : str, Path, or PIL.Image.Image
            Local image path or RGB PIL image.
        top_k : int, optional
            Number of products to return.

        Returns
        -------
        list of RetrievedProduct
            Ranked catalog matches for the image.
        """
        if isinstance(image, Image.Image):
            embedding = self.encoder.encode_images([image.convert("RGB")])
        else:
            embedding = self.encoder.encode_image_path(str(image))

        return self._search_embedding(embedding, top_k)

    def _search_embedding(
        self,
        embedding: np.ndarray,
        top_k: int,
    ) -> List[RetrievedProduct]:
        """
        Run FAISS search for one query embedding.

        Parameters
        ----------
        embedding : np.ndarray
            Shape (1, dim) L2-normalized query vector.
        top_k : int
            Number of neighbors to retrieve.

        Returns
        -------
        list of RetrievedProduct
            Ranked product hits.
        """
        scores, indices = self.index.search(embedding.astype("float32"), top_k)
        results: List[RetrievedProduct] = []

        for rank, (score, row_index) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if row_index < 0:
                continue

            row = self.products.iloc[int(row_index)]
            results.append(
                RetrievedProduct(
                    rank=rank,
                    index=int(row_index),
                    score=float(score),
                    uniq_id=str(row["Uniq Id"]),
                    product_name=str(row["Product Name"]),
                    combined_text=strip_amazon_boilerplate(
                        str(row.get("combined_text", ""))
                    ),
                    image_url=str(row.get("Image", "")).split("|")[0],
                    product_url=str(row.get("Product Url", "")),
                )
            )

        return results
