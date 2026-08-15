import pandas as pd
import numpy as np
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor

df = pd.read_csv("data/processed/clean_products.csv")

index = faiss.read_index("data/processed/products_faiss.index")

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

query = "500 piece jigsaw puzzle"

inputs = processor(
    text=[query],
    return_tensors="pt",
    padding=True,
    truncation=True
)

with torch.no_grad():
    query_embedding = model.get_text_features(**inputs).cpu().numpy().astype("float32")

faiss.normalize_L2(query_embedding)

scores, indices = index.search(query_embedding, 5)

print("Query:", query)
print()

for rank, idx in enumerate(indices[0], start=1):
    print(f"Result {rank}:")
    print(df.iloc[idx]["Product Name"])
    print(df.iloc[idx]["combined_text"][:500])
    print("-" * 80)