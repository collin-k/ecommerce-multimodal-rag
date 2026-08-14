import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

image_embeddings = []

for i in range(10):
    image_path = f"data/images/product_{i}.jpg"
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    image_embeddings.append(embedding.cpu().numpy())

image_embeddings = np.vstack(image_embeddings)

np.save(
    "data/processed/test_image_embeddings.npy",
    image_embeddings
)

print("Image embeddings saved successfully!")
print("Shape:", image_embeddings.shape)