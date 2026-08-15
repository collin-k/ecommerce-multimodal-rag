"""
Legacy entry point for downloading a small product-image sample.

Prefer ``python -m src.image_downloader``.
"""

from .image_downloader import download_product_images


def main() -> None:
    """Download the first 10 product images (original pilot behavior)."""
    saved, skipped, failed = download_product_images(limit=10)
    print(
        f"Done. saved_or_present={len(saved)} skipped_existing={skipped} "
        f"failed={failed}"
    )


if __name__ == "__main__":
    main()
