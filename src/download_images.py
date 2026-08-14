import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

df = pd.read_csv("data/processed/clean_products.csv")

os.makedirs("data/images", exist_ok=True)

for i in range(10):
    image_url = df.iloc[i]["Image"]

    try:
        response = requests.get(image_url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        image.save(f"data/images/product_{i}.jpg")

        print(f"Saved product_{i}.jpg")

    except Exception as e:
        print(f"Failed product {i}: {e}")