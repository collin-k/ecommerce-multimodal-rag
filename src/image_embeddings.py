"""
Embed locally downloaded product images with CLIP.

Prefer this module over ad-hoc scripts. Images are expected at
``data/images/product_{index}.jpg``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .clip_encoder import ClipEncoder
from .config import IMAGE_EMBEDDINGS_PATH, IMAGES_DIR


def collect_image_paths(images_dir: Path = IMAGES_DIR) -> List[Path]:
    """
    List downloaded product images in index order.

    Parameters
    ----------
    images_dir : Path, optional
        Directory containing ``product_*.jpg`` files.

    Returns
    -------
    list of Path
        Sorted image paths.
    """
    return sorted(images_dir.glob("product_*.jpg"))


def embed_downloaded_images(
    images_dir: Path = IMAGES_DIR,
    output_path: Path = IMAGE_EMBEDDINGS_PATH,
) -> Tuple[np.ndarray, List[int]]:
    """
    Encode local product images and save embeddings.

    Parameters
    ----------
    images_dir : Path, optional
        Directory of downloaded JPEGs.
    output_path : Path, optional
        Destination ``.npy`` file.

    Returns
    -------
    tuple
        Embedding matrix and corresponding catalog row indices.
    """
    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No product_*.jpg files found in {images_dir}. "
            "Run: python -m src.image_downloader --limit 100"
        )

    encoder = ClipEncoder()
    images = [Image.open(path).convert("RGB") for path in image_paths]
    embeddings = encoder.encode_images(images)
    row_indices = [int(path.stem.split("_")[1]) for path in image_paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved {embeddings.shape} embeddings to {output_path}")
    return embeddings, row_indices


def main() -> None:
    """Embed all downloaded product images."""
    parser = argparse.ArgumentParser(
        description="Create CLIP embeddings for downloaded product images.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(IMAGES_DIR),
        help="Directory of product_*.jpg files.",
    )
    args = parser.parse_args()
    embed_downloaded_images(images_dir=Path(args.images_dir))


if __name__ == "__main__":
    main()
