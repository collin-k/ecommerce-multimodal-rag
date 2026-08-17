import numpy as np
import faiss
from PIL import Image
import pandas as pd

from .clip_embeddings import ClipEncoder


# Load product data
df = pd.read_csv(
    "data/processed/clean_products.csv"
)


# Load image embeddings created earlier
image_embeddings = np.load(
    "data/processed/image_embeddings.npy"
).astype("float32")


# Load mapping:
# FAISS position -> original dataframe row
row_indices = np.load(
    "data/processed/image_row_indices.npy"
)


print("Image embeddings shape:", image_embeddings.shape)
print("Row mappings:", len(row_indices))


# Normalize stored embeddings
faiss.normalize_L2(image_embeddings)


# Build FAISS index
dimension = image_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(image_embeddings)


# Load CLIP encoder
encoder = ClipEncoder()


# Query image
query_image_path = "data/images/product_0.jpg"

query_image = Image.open(
    query_image_path
).convert("RGB")


# Generate query embedding
query_embedding = encoder.encode_images(
    [query_image]
).astype("float32")


# Normalize query embedding
faiss.normalize_L2(query_embedding)


# Search
top_k = 5

scores, indices = index.search(
    query_embedding,
    top_k
)


print("\nQuery image:")
print(query_image_path)

print("\nTop matches:")


for rank, faiss_idx in enumerate(indices[0]):

    # Map FAISS position back
    # to original dataframe row
    original_row_idx = int(
        row_indices[faiss_idx]
    )

    product = df.iloc[
        original_row_idx
    ]

    print("\n------------------------")
    print("Rank:", rank + 1)

    print(
        "FAISS position:",
        faiss_idx
    )

    print(
        "Original product row:",
        original_row_idx
    )

    print(
        "Similarity score:",
        float(scores[0][rank])
    )

    if "Product Name" in df.columns:
        print(
            "Product:",
            product["Product Name"]
        )

    if "Category" in df.columns:
        print(
            "Category:",
            product["Category"]
        )

    if "Selling Price" in df.columns:
        print(
            "Price:",
            product["Selling Price"]
        )