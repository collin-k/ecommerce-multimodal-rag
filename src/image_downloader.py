"""
Download product images from the cleaned Amazon catalog.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import requests
from PIL import Image

from .config import (
    DEFAULT_IMAGE_DOWNLOAD_LIMIT,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_TIMEOUT_SECONDS,
    DOWNLOAD_WORKERS,
    IMAGES_DIR,
    PRODUCTS_CSV,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


MAX_URLS_PER_PRODUCT = 3


def image_urls(image_field: object) -> List[str]:
    """
    Split a pipe-separated Image column into URL strings.

    Parameters
    ----------
    image_field : object
        Raw Image field that may contain multiple URLs.

    Returns
    -------
    list of str
        Non-empty URLs in catalog order.
    """
    if not isinstance(image_field, str) or not image_field.strip():
        return []
    return [part.strip() for part in image_field.split("|") if part.strip()]


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
    urls = image_urls(image_field)
    return urls[0] if urls else ""


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


def _download_row(
    products: pd.DataFrame,
    row_index: int,
    output_dir: Path,
    timeout_seconds: int,
    max_retries: int,
) -> Tuple[str, Optional[Path], str]:
    """
    Download one catalog row's primary photo, trying fallback URLs.

    Parameters
    ----------
    products : pandas.DataFrame
        Cleaned catalog.
    row_index : int
        Catalog row to fetch.
    output_dir : Path
        Directory for ``product_{index}.jpg``.
    timeout_seconds : int
        HTTP timeout per attempt.
    max_retries : int
        Attempts per URL.

    Returns
    -------
    tuple
        Status (``saved``, ``skipped``, or ``failed``), destination path
        when present, and an error message on failure.
    """
    if row_index < 0 or row_index >= len(products):
        return "failed", None, f"row {row_index} out of range"

    destination = output_dir / f"product_{int(row_index)}.jpg"
    if destination.exists():
        return "skipped", destination, ""

    urls = image_urls(products.iloc[int(row_index)].get("Image", ""))[:MAX_URLS_PER_PRODUCT]
    if not urls:
        return "failed", None, "no image URL"

    last_error = "download failed"
    for image_url in urls:
        try:
            _download_one_image(
                image_url=image_url,
                destination=destination,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            return "saved", destination, ""
        except Exception as error:
            last_error = str(error)
    return "failed", None, last_error


def download_product_images(
    products_csv: Path = PRODUCTS_CSV,
    output_dir: Path = IMAGES_DIR,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
    timeout_seconds: int = DOWNLOAD_TIMEOUT_SECONDS,
    max_retries: int = DOWNLOAD_MAX_RETRIES,
    workers: int = DOWNLOAD_WORKERS,
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
        ``None`` downloads the entire catalog.
    indices : sequence of int or None, optional
        Explicit row indices to download.
    timeout_seconds : int, optional
        HTTP timeout per image request.
    max_retries : int, optional
        Retry count for failed downloads.
    workers : int, optional
        Parallel download threads.

    Returns
    -------
    tuple
        Saved paths, skip count, and failure count.
    """
    products = pd.read_csv(products_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        row_count = len(products) if limit is None else min(limit, len(products))
        row_indices = list(range(row_count))
    else:
        row_indices = list(indices)

    saved_paths: List[Path] = []
    skipped = 0
    failed = 0
    total = len(row_indices)
    worker_count = max(1, workers)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(
                _download_row,
                products,
                row_index,
                output_dir,
                timeout_seconds,
                max_retries,
            )
            for row_index in row_indices
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            status, path, error = future.result()
            if status == "skipped" and path is not None:
                saved_paths.append(path)
                skipped += 1
            elif status == "saved" and path is not None:
                saved_paths.append(path)
            else:
                failed += 1
                if failed <= 10 and error:
                    print(f"Failed product: {error}")
            if done % 100 == 0 or done == total:
                print(
                    f"Progress {done}/{total} "
                    f"saved_or_present={len(saved_paths)} "
                    f"skipped_existing={skipped} failed={failed}",
                    flush=True,
                )

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
        help="Leading catalog rows to download. Default: entire catalog.",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
        help="Parallel download threads.",
    )
    args = parser.parse_args()

    selected_indices = parse_indices(args.indices) if args.indices else None
    saved, skipped, failed = download_product_images(
        limit=None if selected_indices is not None else args.limit,
        indices=selected_indices,
        timeout_seconds=args.timeout,
        workers=args.workers,
    )

    print(
        f"Done. saved_or_present={len(saved)} skipped_existing={skipped} "
        f"failed={failed} dir={IMAGES_DIR}"
    )


if __name__ == "__main__":
    main()
