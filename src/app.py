"""
Streamlit shop assistant for multimodal product RAG.

Run from the project root:

    streamlit run src/app.py

``.streamlit/config.toml`` disables Streamlit's file watcher so it does
not probe unused ``transformers`` vision models.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

try:
    from .config import DEFAULT_TOP_K, IMAGES_DIR
    from .rag import ProductRagAssistant, is_show_image_question, require_api_key
    from .retriever import ProductRetriever, RetrievedProduct, strip_amazon_boilerplate
except ImportError:
    from config import DEFAULT_TOP_K, IMAGES_DIR
    from rag import ProductRagAssistant, is_show_image_question, require_api_key
    from retriever import ProductRetriever, RetrievedProduct, strip_amazon_boilerplate

SUGGESTED_PROMPTS = (
    "What are the features of the DB Longboards CoreFlex Crossbow?",
    "Show me a picture of the Woodstock collage 500 piece puzzle.",
    "Compare the LEGO Minecraft Creeper set with the LEGO Friends Heartlake Surf Shop.",
)

PAGE_STYLE = """
<style>
    .stApp { background-color: #f7f1e8; }
    [data-testid="stHeader"] { background: #f7f1e8; }
    [data-testid="stToolbar"] { display: none; }
    .stAppDeployButton { display: none; }
    .hero-title {
        font-size: 2rem;
        font-weight: 650;
        color: #1b3a4b;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .hero-sub { color: #5c6b73; margin-bottom: 1.2rem; }
    .product-name { font-weight: 600; color: #1b3a4b; }
    .product-price { color: #8a6d2b; font-weight: 600; }
    .stChatInput textarea { background: #fffdf8; }
</style>
"""


st.set_page_config(
    page_title="Shop assistant",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the product catalog…")
def load_retriever() -> ProductRetriever:
    """
    Load the shared product retriever once per session.

    Returns
    -------
    ProductRetriever
        Cached multimodal retriever.
    """
    return ProductRetriever()


@st.cache_resource(show_spinner="Starting the assistant…")
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


def product_price(product: RetrievedProduct) -> str:
    """
    Extract a selling price from catalog text when present.

    Parameters
    ----------
    product : RetrievedProduct
        Ranked catalog hit.

    Returns
    -------
    str
        Price string, or empty if missing.
    """
    for line in product.combined_text.splitlines():
        if not line.startswith("Selling Price:"):
            continue
        price = line.split(":", 1)[1].strip()
        if price and "total price" not in price.lower():
            return price
    return ""


def product_blurb(product: RetrievedProduct, limit: int = 160) -> str:
    """
    Return a short shopper-facing snippet for a product card.

    Parameters
    ----------
    product : RetrievedProduct
        Ranked catalog hit.
    limit : int, optional
        Maximum character length.

    Returns
    -------
    str
        Truncated about-text or category line.
    """
    about = ""
    category = ""
    for line in product.combined_text.splitlines():
        if line.startswith("About Product:"):
            about = line.split(":", 1)[1].strip()
        elif line.startswith("Category:"):
            category = line.split(":", 1)[1].strip().split("|")[-1].strip()
    text = about or category or product.product_name
    text = strip_amazon_boilerplate(text)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def product_image_source(product: RetrievedProduct) -> Optional[str]:
    """
    Prefer a local JPEG, then a remote catalog URL.

    Parameters
    ----------
    product : RetrievedProduct
        Ranked catalog hit.

    Returns
    -------
    str or None
        Path or URL Streamlit can render.
    """
    local_image = local_product_image(product)
    if local_image is not None:
        return str(local_image)
    if product.image_url.startswith("http"):
        return product.image_url
    return None


def render_product_card(
    product: RetrievedProduct,
    show_scores: bool,
    card_key: str,
) -> None:
    """
    Render one product as a compact shop card.

    Parameters
    ----------
    product : RetrievedProduct
        Ranked catalog hit.
    show_scores : bool
        When True, show cosine similarity for testers.
    card_key : str
        Unique widget key so Amazon links do not collide across turns.
    """
    image_source = product_image_source(product)
    if image_source is not None:
        st.image(image_source, width=220)
    st.markdown(
        f"<div class='product-name'>{product.product_name}</div>",
        unsafe_allow_html=True,
    )
    price = product_price(product)
    if price:
        st.markdown(
            f"<div class='product-price'>{price}</div>",
            unsafe_allow_html=True,
        )
    st.caption(product_blurb(product))
    if product.product_url:
        st.link_button(
            "View on Amazon",
            product.product_url,
            use_container_width=True,
            key=f"amazon-{card_key}",
        )
    if show_scores:
        st.caption(f"Match score {product.score:.3f}")


def render_product_grid(
    products: List[RetrievedProduct],
    show_scores: bool,
    lead_with_image: bool = False,
    grid_key: str = "grid",
) -> None:
    """
    Display retrieved products as a card grid.

    Parameters
    ----------
    products : list of RetrievedProduct
        Ranked retrieval hits.
    show_scores : bool
        Show tester scores on each card.
    lead_with_image : bool, optional
        When True, enlarge the first product image (show-me requests).
    grid_key : str, optional
        Prefix for widget keys so cards stay unique across chat turns.
    """
    if not products:
        st.info("I could not find a matching product in this catalog.")
        return

    if lead_with_image:
        hero = product_image_source(products[0])
        if hero is not None:
            st.image(hero, caption=products[0].product_name, width=360)

    columns = st.columns(min(3, len(products)))
    for index, product in enumerate(products):
        with columns[index % len(columns)]:
            render_product_card(
                product,
                show_scores=show_scores,
                card_key=f"{grid_key}-{index}-{product.index}",
            )

    if show_scores:
        with st.expander("Why this match?"):
            for product in products:
                st.markdown(
                    f"**{product.rank}. {product.product_name}** "
                    f"({product.score:.3f})"
                )
                st.write(product.combined_text[:800])


def render_sidebar() -> Dict[str, Any]:
    """
    Draw shopper chrome plus collapsed developer controls.

    Returns
    -------
    dict
        ``top_k``, ``retrieve_only``, ``show_scores``, and the upload.
    """
    with st.sidebar:
        st.markdown("### Shop assistant")
        st.caption("Ask in plain language or attach a product photo.")
        uploaded = st.file_uploader(
            "Attach a photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="chat_image",
        )
        if uploaded is not None:
            st.image(uploaded, width=160)
            if st.button(
                "Identify this photo",
                use_container_width=True,
                key="identify-photo",
            ):
                st.session_state.queued_prompt = (
                    "Identify the product in this image and describe how it is used."
                )
                st.rerun()
        with st.expander("Developer", expanded=False):
            top_k = st.slider(
                "Products to retrieve",
                min_value=1,
                max_value=10,
                value=DEFAULT_TOP_K,
            )
            retrieve_only = st.toggle("Retrieve only (no LLM)", value=False)
            show_scores = st.toggle("Show match scores", value=False)
        return {
            "top_k": top_k,
            "retrieve_only": retrieve_only,
            "show_scores": show_scores,
            "uploaded": uploaded,
        }


def render_user_message(message: Dict[str, Any]) -> None:
    """
    Render a shopper chat turn.

    Parameters
    ----------
    message : dict
        Stored user message with optional image bytes.
    """
    with st.chat_message("user"):
        if message.get("image_bytes"):
            st.image(message["image_bytes"], width=220)
        if message.get("text"):
            st.write(message["text"])


def render_assistant_message(
    message: Dict[str, Any],
    show_scores: bool,
    message_key: str,
) -> None:
    """
    Render an assistant turn with answer text and product cards.

    Parameters
    ----------
    message : dict
        Stored assistant payload.
    show_scores : bool
        Tester score visibility.
    message_key : str
        Unique prefix for product-card widgets in this turn.
    """
    with st.chat_message("assistant"):
        if message.get("text"):
            st.write(message["text"])
        render_product_grid(
            message.get("products") or [],
            show_scores=show_scores,
            lead_with_image=bool(message.get("lead_with_image")),
            grid_key=message_key,
        )


def answer_turn(
    question: str,
    image: Optional[Image.Image],
    top_k: int,
    retrieve_only: bool,
) -> Dict[str, Any]:
    """
    Run retrieval or RAG for one chat turn.

    Parameters
    ----------
    question : str
        User text. May be empty when an image is attached.
    image : PIL.Image.Image or None
        Optional uploaded product photo.
    top_k : int
        Retrieval depth.
    retrieve_only : bool
        Skip the LLM and return catalog matches only.

    Returns
    -------
    dict
        Assistant message payload.
    """
    retriever = load_retriever()
    prompt = question.strip()
    if image is not None and not prompt:
        prompt = "Identify the product in this image and describe how it is used."

    if retrieve_only:
        with st.spinner("Finding products…"):
            if image is not None:
                products = retriever.search_image(image, top_k=top_k)
            else:
                products = retriever.search_text(prompt, top_k=top_k)
        text = "Here are the closest matches in the catalog."
        if not products:
            text = "I could not find a matching product in this catalog."
        return {
            "role": "assistant",
            "text": text,
            "products": products,
            "lead_with_image": is_show_image_question(prompt),
        }

    try:
        require_api_key()
        assistant = load_assistant()
    except Exception as error:
        return {
            "role": "assistant",
            "text": (
                "I need an API key to write an answer. "
                "You can still search the catalog: open Developer in the sidebar "
                "and turn on Retrieve only."
            ),
            "products": [],
            "lead_with_image": False,
            "error": str(error),
        }

    try:
        with st.spinner("Writing an answer…"):
            if image is not None:
                result = assistant.answer_image(prompt, image, top_k=top_k)
            else:
                result = assistant.answer_text(prompt, top_k=top_k)
    except Exception as error:
        return {
            "role": "assistant",
            "text": (
                "The language model did not return an answer. "
                f"{error}"
            ),
            "products": [],
            "lead_with_image": False,
        }

    answer = str(result.get("answer") or "").strip()
    products = list(result.get("products") or [])
    if not answer and products:
        top = products[0]
        answer = (
            f"The closest catalog match is {top.product_name}. "
            "Here are the retrieved products."
        )
    if not answer:
        answer = "I could not find a matching product in this catalog."

    return {
        "role": "assistant",
        "text": answer,
        "products": products,
        "lead_with_image": is_show_image_question(prompt),
    }


def main() -> None:
    """Run the shopper-facing Streamlit chatbot."""
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    settings = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = ""

    if "catalog_ready" not in st.session_state:
        with st.spinner(
            "Getting the catalog ready. The first load can take a minute…"
        ):
            try:
                load_retriever()
            except Exception as error:
                st.error(
                    "I could not load the product catalog. "
                    "Check that the FAISS index and embeddings are present."
                )
                st.caption(str(error))
                return
        st.session_state.catalog_ready = True

    st.markdown("<div class='hero-title'>Shop assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Ask about products in this Amazon catalog, "
        "or attach a photo to identify something.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.caption("Try one of these:")
        chip_columns = st.columns(len(SUGGESTED_PROMPTS))
        for chip_index, (column, suggestion) in enumerate(
            zip(chip_columns, SUGGESTED_PROMPTS)
        ):
            with column:
                if st.button(
                    suggestion,
                    use_container_width=True,
                    key=f"suggest-{chip_index}",
                ):
                    st.session_state.queued_prompt = suggestion
                    st.rerun()

    for turn_index, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            render_user_message(message)
        else:
            render_assistant_message(
                message,
                show_scores=settings["show_scores"],
                message_key=f"turn-{turn_index}",
            )

    typed = st.chat_input("Ask about a product…")
    uploaded = settings["uploaded"]
    prompt = (typed or st.session_state.queued_prompt or "").strip()
    st.session_state.queued_prompt = ""
    upload_id = None
    if uploaded is not None:
        upload_id = f"{uploaded.name}:{int(uploaded.size)}"
        if not prompt and upload_id != st.session_state.get("last_image_id"):
            prompt = "Identify the product in this image and describe how it is used."
    if not prompt:
        return
    if upload_id is not None:
        st.session_state.last_image_id = upload_id

    image = None
    image_bytes = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

    user_message = {"role": "user", "text": prompt, "image_bytes": image_bytes}
    st.session_state.messages.append(user_message)

    try:
        with st.spinner("Looking through the catalog…"):
            assistant_message = answer_turn(
                prompt,
                image,
                top_k=int(settings["top_k"]),
                retrieve_only=bool(settings["retrieve_only"]),
            )
    except Exception as error:
        assistant_message = {
            "role": "assistant",
            "text": (
                "Something went wrong while searching the catalog. "
                f"{error}"
            ),
            "products": [],
            "lead_with_image": False,
        }

    st.session_state.messages.append(assistant_message)
    st.rerun()


main()
