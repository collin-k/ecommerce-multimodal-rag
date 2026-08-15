"""
Download product images from the cleaned Amazon catalog.
"""

from __future__ import annotations

import argparse
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from PIL import Image

from .config import (
    DEFAULT_IMAGE_DOWNLOAD_LIMIT,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_TIMEOUT_SECONDS,
    IMAGES_DIR,
    PRODUCTS_CSV,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def first_image_url(image_field: object) -> str:
    """
    Extract the first URL from a pipe-separated Image column value.

    Parameters
    ----------
    image_field : object
        Raw Image field that may contain multiple URLs.

    Returns
    -------
    str
        First URL, or an empty string when missing.
    """
    if not isinstance(image_field, str) or not image_field.strip():
        return ""
    return image_field.split("|")[0].strip()


def parse_indices(raw_indices: str) -> List[int]:
    """
    Parse a comma-separated list of row indices.

    Parameters
    ----------
    raw_indices : str
        Values such as ``"0,5,12"``.

    Returns
    -------
    list of int
        Parsed row indices.
    """
    parts = [part.strip() for part in raw_indices.split(",") if part.strip()]
    return [int(part) for part in parts]


def _download_one_image(
    image_url: str,
    destination: Path,
    timeout_seconds: int,
    max_retries: int,
) -> None:
    """
    Fetch one image URL and save it as JPEG.

    Parameters
    ----------
    image_url : str
        Remote product image URL.
    destination : Path
        Local JPEG path.
    timeout_seconds : int
        HTTP timeout per attempt.
    max_retries : int
        Number of attempts before failing.
    """
    last_error: Optional[Exception] = None
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                image_url,
                timeout=timeout_seconds,
                headers=headers,
            )
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(destination, format="JPEG")
            return
        except Exception as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(0.5 * attempt)

    raise RuntimeError(str(last_error))


def download_product_images(
    products_csv: Path = PRODUCTS_CSV,
    output_dir: Path = IMAGES_DIR,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_retries: int = DOWNLOAD_MAX_RETRIES,
) -> Tuple[List[Path], int, int]:
    """
    Download product images and save them as ``product_{index}.jpg``.

    Parameters
    ----------
    products_csv : Path, optional
        Cleaned product catalog.
    output_dir : Path, optional
        Directory for downloaded JPEG files.
    limit : int or None, optional
        Download the first ``limit`` rows when ``indices`` is omitted.
    indices : sequence of int or None, optional
        Explicit row indices to download.
    timeout_seconds : int, optional
        HTTP timeout per image request.
    max_retries : int, optional
        Retry count for failed downloads.

    Returns
    -------
    tuple
        Saved paths, skip count, and failure count.
    """
    products = pd.read_csv(products_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        row_count = len(products) if limit is None else min(limit, len(products))
        row_indices: Iterable[int] = range(row_count)
    else:
        row_indices = indices

    saved_paths: List[Path] = []
    skipped = 0
    failed = 0

    for row_index in row_indices:
        if row_index < 0 or row_index >= len(products):
            print(f"Skipped product {row_index}: index out of range")
            failed += 1
            continue

        destination = output_dir / f"product_{int(row_index)}.jpg"
        if destination.exists():
            saved_paths.append(destination)
            skipped += 1
            print(f"Skipped {destination.name} (already exists)")
            continue

        image_url = first_image_url(products.iloc[int(row_index)].get("Image", ""))
        if not image_url:
            print(f"Failed product {row_index}: missing image URL")
            failed += 1
            continue

        try:
            _download_one_image(
                image_url=image_url,
                destination=destination,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            saved_paths.append(destination)
            print(f"Saved {destination.name}")
        except Exception as error:
            failed += 1
            print(f"Failed product {row_index}: {error}")

    return saved_paths, skipped, failed


def main() -> None:
    """Download product images for multimodal demos and evaluation."""
    parser = argparse.ArgumentParser(
        description="Download product images from the cleaned catalog.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_IMAGE_DOWNLOAD_LIMIT,
        help="Number of leading catalog rows to download (ignored with --indices).",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated row indices, e.g. 0,5,12",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DOWNLOAD_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds per request.",
    )
    args = parser.parse_args()

    selected_indices = parse_indices(args.indices) if args.indices else None
    saved, skipped, failed = download_product_images(
        limit=None if selected_indices is not None else args.limit,
        indices=selected_indices,
        timeout_seconds=args.timeout,
    )

    print(
        f"Done. saved_or_present={len(saved)} skipped_existing={skipped} "
        f"failed={failed} dir={IMAGES_DIR}"
    )


if __name__ == "__main__":
    main()
