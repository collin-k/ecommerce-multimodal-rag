import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

image_embeddings = np.load(
    "data/processed/test_image_embeddings.npy"
).astype("float32")

faiss.normalize_L2(image_embeddings)

index = faiss.IndexFlatIP(512)
index.add(image_embeddings)

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

query_image = Image.open(
    "data/images/product_0.jpg"
).convert("RGB")

inputs = processor(
    images=query_image,
    return_tensors="pt"
)

with torch.no_grad():
    query_embedding = model.get_image_features(**inputs)

query_embedding = query_embedding.cpu().numpy().astype("float32")

faiss.normalize_L2(query_embedding)

scores, indices = index.search(query_embedding, 5)

print("Top matches:")
print(indices[0])

print("Similarity scores:")
print(scores[0])