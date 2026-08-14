import pandas as pd

df = pd.read_csv("data/processed/clean_products.csv")

matches = df[
    df["Product Name"]
    .fillna("")
    .str.contains("puzzle|game|skateboard", case=False, regex=True)
]

print("Matching products:", len(matches))
print(matches["Product Name"].head(30).to_string(index=False))