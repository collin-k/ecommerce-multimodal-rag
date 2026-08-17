"""
Embed locally downloaded product images with CLIP.

Images are expected at ``data/images/product_{index}.jpg``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    from .clip_encoder import ClipEncoder
    from .config import IMAGE_EMBEDDINGS_PATH, IMAGE_ROW_INDICES_PATH, IMAGES_DIR
except ImportError:
    from clip_encoder import ClipEncoder
    from config import IMAGE_EMBEDDINGS_PATH, IMAGE_ROW_INDICES_PATH, IMAGES_DIR


def collect_image_paths(images_dir: Path = IMAGES_DIR) -> List[Path]:
    """
    Return downloaded product images in filename order.

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
    encoder: ClipEncoder | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode product photos and save embeddings plus catalog-row ids.

    Parameters
    ----------
    images_dir : Path, optional
        Directory of downloaded JPEGs.
    output_path : Path, optional
        Destination ``.npy`` file for embeddings.
    encoder : ClipEncoder or None, optional
        Shared CLIP encoder. Created on demand when omitted.

    Returns
    -------
    tuple
        Embedding matrix and corresponding catalog row indices.
    """
    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No product_*.jpg files found in {images_dir}. "
            "Run: python -m src.image_downloader"
        )

    print(f"Found {len(image_paths)} product images")
    clip = encoder or ClipEncoder()
    embedding_batches: List[np.ndarray] = []
    row_indices: List[int] = []
    batch_size = 32
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images: List[Image.Image] = []
        batch_rows: List[int] = []
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                batch_rows.append(int(path.stem.split("_")[1]))
            except Exception as error:
                print(f"Skipping {path.name}: {error}")
        if not images:
            continue
        embedding_batches.append(clip.encode_images(images))
        row_indices.extend(batch_rows)
        for image in images:
            image.close()
        done = min(start + batch_size, len(image_paths))
        if done % 160 == 0 or done == len(image_paths):
            print(f"Embedded {done} / {len(image_paths)}", flush=True)

    if not embedding_batches:
        raise RuntimeError("No image embeddings were successfully generated.")

    embeddings = np.vstack(embedding_batches)
    row_map = np.asarray(row_indices, dtype=int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    indices_path = IMAGE_ROW_INDICES_PATH
    if output_path != IMAGE_EMBEDDINGS_PATH:
        indices_path = output_path.with_name("image_row_indices.npy")
    np.save(indices_path, row_map)

    print(f"Saved {embeddings.shape} embeddings to {output_path}")
    print(f"Saved {len(row_map)} row mappings to {indices_path}")
    return embeddings, row_map


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
