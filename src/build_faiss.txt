import numpy as np
import faiss

embeddings = np.load("data/processed/text_embeddings.npy").astype("float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "data/processed/products_faiss.index")

print("FAISS index created successfully!")
print("Number of vectors:", index.ntotal)
print("Embedding dimension:", dimension)