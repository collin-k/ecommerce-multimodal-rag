import pandas as pd
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor

df = pd.read_csv("data/processed/clean_products.csv")

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

all_embeddings = []
batch_size = 32

for start in range(0, len(df), batch_size):
    end = start + batch_size
    texts = df["clip_text"].iloc[start:end].fillna("").tolist()

    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)

    all_embeddings.append(embeddings.cpu().numpy())

    print(f"Processed {min(end, len(df))} / {len(df)} products")

text_embeddings = np.vstack(all_embeddings)

np.save("data/processed/text_embeddings.npy", text_embeddings)

print("Saved new CLIP embeddings successfully!")
print("Final shape:", text_embeddings.shape)