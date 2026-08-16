import numpy as np
import faiss

embeddings = np.load("data/processed/text_embeddings.npy").astype("float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "data/processed/products_faiss.index")
print(f"Wrote data/processed/products_faiss.index ({index.ntotal} x {dimension})")