"""
Streamlit UI for the multimodal e-commerce product chatbot.

Run from the project root:

    streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import streamlit as st
from PIL import Image

try:
    from .config import DEFAULT_TOP_K, IMAGES_DIR
    from .rag import ProductRagAssistant, require_api_key
    from .retriever import ProductRetriever, RetrievedProduct
except ImportError:
    from config import DEFAULT_TOP_K, IMAGES_DIR
    from rag import ProductRagAssistant, require_api_key
    from retriever import ProductRetriever, RetrievedProduct


st.set_page_config(
    page_title="E-commerce Multimodal RAG",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading CLIP + FAISS retriever...")
def load_retriever() -> ProductRetriever:
    """
    Load the shared product retriever once per session.

    Returns
    -------
    ProductRetriever
        Cached multimodal retriever.
    """
    return ProductRetriever()


@st.cache_resource(show_spinner="Loading RAG assistant...")
def load_assistant() -> ProductRagAssistant:
    """
    Load the RAG assistant once per session.

    Returns
    -------
    ProductRagAssistant
        Cached retrieval-augmented assistant.
    """
    return ProductRagAssistant(retriever=load_retriever())


def local_product_image(product: RetrievedProduct) -> Optional[Path]:
    """
    Return a local JPEG path for a retrieved product when it exists.

    Parameters
    ----------
    product : RetrievedProduct
        Ranked catalog hit.

    Returns
    -------
    Path or None
        ``data/images/product_{index}.jpg`` if present.
    """
    image_path = IMAGES_DIR / f"product_{product.index}.jpg"
    if image_path.is_file():
        return image_path
    return None


def render_products(products: List[RetrievedProduct]) -> None:
    """
    Display retrieved products with snippets, links, and images.

    Parameters
    ----------
    products : list of RetrievedProduct
        Ranked retrieval hits to show.
    """
    st.subheader("Retrieved products")
    for product in products:
        with st.expander(
            f"{product.rank}. {product.product_name} ({product.score:.3f})",
            expanded=product.rank == 1,
        ):
            st.write(product.combined_text[:1200])
            if product.product_url:
                st.markdown(f"[Product page]({product.product_url})")

            local_image = local_product_image(product)
            if local_image is not None:
                st.image(str(local_image), width=220)
            elif product.image_url and product.image_url.startswith("http"):
                st.image(product.image_url, width=220)


def main() -> None:
    """Run the Streamlit multimodal chatbot app."""
    st.title("Multimodal E-commerce Assistant")
    st.caption(
        "Ask product questions with text, upload an image, or both. "
        "Answers are grounded in CLIP + FAISS retrieval over the Amazon catalog."
    )

    mode = st.radio(
        "Query mode",
        options=["Text question", "Image question", "Retrieve only"],
        horizontal=True,
    )
    question = st.text_input(
        "Your question",
        placeholder="What are the features of this product?",
    )
    uploaded = st.file_uploader(
        "Upload a product image (required for image mode)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    top_k = st.slider(
        "Products to retrieve",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    if not st.button("Ask", type="primary"):
        return

    try:
        retriever = load_retriever()
    except Exception as error:
        st.error(f"Could not load retriever: {error}")
        return

    image: Optional[Image.Image] = None
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=280)

    try:
        if mode == "Retrieve only":
            if image is not None:
                products = retriever.search_image(image, top_k=top_k)
            elif question.strip():
                products = retriever.search_text(question, top_k=top_k)
            else:
                st.warning("Enter a question or upload an image.")
                return
            render_products(products)
            return

        try:
            require_api_key()
            assistant = load_assistant()
        except Exception as error:
            st.error(str(error))
            st.info("Use Retrieve only mode to search the catalog without an API key.")
            return

        if mode == "Image question":
            if image is None:
                st.warning("Upload an image for image questions.")
                return
            prompt = question.strip() or (
                "Identify the product in this image and describe how it is used."
            )
            result = assistant.answer_image(prompt, image, top_k=top_k)
        else:
            if not question.strip():
                st.warning("Enter a text question.")
                return
            if image is not None:
                result = assistant.answer_image(question, image, top_k=top_k)
            else:
                result = assistant.answer_text(question, top_k=top_k)

        st.subheader("Answer")
        st.write(result["answer"])
        render_products(result["products"])

    except Exception as error:
        st.error(str(error))


main()
