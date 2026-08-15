"""
CLIP encoder for text and image embeddings in a shared vector space.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_MODEL_NAME


class ClipEncoder:
    """Encode text and images with a pretrained CLIP model."""

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """
        Load CLIP weights and processor.

        Parameters
        ----------
        model_name : str, optional
            Hugging Face model id for CLIP.
        device : str or None, optional
            Torch device string. Defaults to CUDA when available.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @staticmethod
    def _as_embedding_tensor(features: object) -> torch.Tensor:
        """
        Convert CLIP feature outputs to a 2D embedding tensor.

        Newer ``transformers`` versions return ``BaseModelOutputWithPooling``
        from ``get_text_features`` / ``get_image_features``, with the projected
        embedding stored on ``pooler_output``. Older versions return a tensor.

        Parameters
        ----------
        features : object
            Model output from CLIP feature helpers.

        Returns
        -------
        torch.Tensor
            Embedding tensor with shape (batch, dim).
        """
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        raise TypeError(
            f"Unexpected CLIP feature type: {type(features)!r}"
        )

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a batch of text strings.

        Parameters
        ----------
        texts : sequence of str
            Input strings to encode.
        batch_size : int, optional
            Number of texts per forward pass.

        Returns
        -------
        np.ndarray
            L2-normalized embeddings with shape (n_texts, dim).
        """
        embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = [
                text if isinstance(text, str) else ""
                for text in texts[start : start + batch_size]
            ]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                features = self._as_embedding_tensor(
                    self.model.get_text_features(**inputs)
                )

            embeddings.append(features.cpu().numpy())

        return self._normalize(np.vstack(embeddings))

    def encode_images(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Embed a batch of PIL images.

        Parameters
        ----------
        images : sequence of PIL.Image.Image
            RGB images to encode.
        batch_size : int, optional
            Number of images per forward pass.

        Returns
        -------
        np.ndarray
            L2-normalized embeddings with shape (n_images, dim).
        """
        embeddings: List[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                features = self._as_embedding_tensor(
                    self.model.get_image_features(**inputs)
                )

            embeddings.append(features.cpu().numpy())

        return self._normalize(np.vstack(embeddings))

    def encode_image_path(self, image_path: Union[str, bytes]) -> np.ndarray:
        """
        Embed a single image file.

        Parameters
        ----------
        image_path : str
            Path to an image on disk.

        Returns
        -------
        np.ndarray
            L2-normalized embedding with shape (1, dim).
        """
        image = Image.open(image_path).convert("RGB")
        return self.encode_images([image])

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embedding rows for cosine similarity via inner product.

        Parameters
        ----------
        embeddings : np.ndarray
            Raw embedding matrix.

        Returns
        -------
        np.ndarray
            Float32 matrix with unit-norm rows.
        """
        vectors = embeddings.astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return vectors / norms
