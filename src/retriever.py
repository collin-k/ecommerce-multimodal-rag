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


def _load_image_index(
    embeddings_path: Path,
    row_indices_path: Path,
    n_products: int,
):
    """
    Build an in-memory FAISS index from saved product-image embeddings.

    Parameters
    ----------
    embeddings_path : Path
        ``.npy`` matrix of CLIP image vectors.
    row_indices_path : Path
        Catalog row for each embedding row.
    n_products : int
        Number of rows in the product table.

    Returns
    -------
    tuple
        Image FAISS index and integer catalog-row mapping.

    Raises
    ------
    FileNotFoundError
        If either artifact is missing.
    ValueError
        If shapes or row ids are inconsistent.
    """
    import faiss

    if not embeddings_path.is_file() or not row_indices_path.is_file():
        raise FileNotFoundError(
            "Image FAISS artifacts not found. Run: python -m src.image_embeddings"
        )

    embeddings = np.load(embeddings_path).astype("float32")
    faiss.normalize_L2(embeddings)
    image_index = faiss.IndexFlatIP(embeddings.shape[1])
    image_index.add(embeddings)
    row_map = np.load(row_indices_path).astype(int)

    if len(row_map) != image_index.ntotal:
        raise ValueError(
            f"Image row map ({len(row_map)}) does not match "
            f"image FAISS vectors ({image_index.ntotal})."
        )
    if np.any(row_map < 0) or np.any(row_map >= n_products):
        raise ValueError("Image row map contains catalog indexes outside the table.")

    return image_index, row_map


# Import torch-backed CLIP before FAISS. Loading FAISS first can segfault on some
# macOS / conda OpenMP stacks when CLIP is initialized afterward.
try:
    from .clip_encoder import ClipEncoder
    from .config import (
        DEFAULT_TOP_K,
        FAISS_INDEX_PATH,
        IMAGE_EMBEDDINGS_PATH,
        IMAGE_ROW_INDICES_PATH,
        PRODUCTS_CSV,
    )
    from .query_rewriter import rewrite_clip_query
except ImportError:
    from clip_encoder import ClipEncoder
    from config import (
        DEFAULT_TOP_K,
        FAISS_INDEX_PATH,
        IMAGE_EMBEDDINGS_PATH,
        IMAGE_ROW_INDICES_PATH,
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
        image_embeddings_path: Path = IMAGE_EMBEDDINGS_PATH,
        image_row_indices_path: Path = IMAGE_ROW_INDICES_PATH,
        encoder: Optional[ClipEncoder] = None,
    ) -> None:
        """
        Load the product table, FAISS indexes, and CLIP encoder.

        Parameters
        ----------
        products_csv : Path, optional
            Cleaned product catalog CSV.
        faiss_index_path : Path, optional
            FAISS index built from CLIP text embeddings.
        image_embeddings_path : Path, optional
            CLIP embeddings for downloaded product photos.
        image_row_indices_path : Path, optional
            Catalog row for each vector in the image index.
        encoder : ClipEncoder or None, optional
            Shared CLIP encoder. Created on demand when omitted.
        """
        self.encoder = encoder or ClipEncoder()
        import faiss

        faiss.omp_set_num_threads(1)
        self.products = pd.read_csv(products_csv)
        self.index = faiss.read_index(str(faiss_index_path))
        self.image_index, self.image_row_indices = _load_image_index(
            embeddings_path=image_embeddings_path,
            row_indices_path=image_row_indices_path,
            n_products=len(self.products),
        )

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
        return self._search_embedding(embedding, top_k, self.index)

    def search_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[RetrievedProduct]:
        """
        Retrieve products for an uploaded product image.

        Searches the product-photo FAISS index (downloaded JPEGs), not the
        text-product index.

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

        return self._search_embedding(
            embedding,
            top_k,
            self.image_index,
            self.image_row_indices,
        )

    def _search_embedding(
        self,
        embedding: np.ndarray,
        top_k: int,
        index,
        row_map: Optional[np.ndarray] = None,
    ) -> List[RetrievedProduct]:
        """
        Run FAISS search for one query embedding.

        Parameters
        ----------
        embedding : np.ndarray
            Shape (1, dim) L2-normalized query vector.
        top_k : int
            Number of neighbors to retrieve.
        index
            FAISS index to search.
        row_map : ndarray or None, optional
            Maps FAISS ids to catalog rows. Identity when omitted.

        Returns
        -------
        list of RetrievedProduct
            Ranked product hits.
        """
        depth = min(top_k, index.ntotal)
        scores, indices = index.search(embedding.astype("float32"), depth)
        results: List[RetrievedProduct] = []

        for rank, (score, faiss_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if faiss_id < 0:
                continue

            row_index = (
                int(row_map[int(faiss_id)])
                if row_map is not None
                else int(faiss_id)
            )
            row = self.products.iloc[row_index]
            results.append(
                RetrievedProduct(
                    rank=rank,
                    index=row_index,
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
