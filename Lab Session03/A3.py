import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
file_path = "/content/Lab Session Data.xlsx"  
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Open"] = pd.to_numeric(df["Open"], errors="coerce")

df = df.dropna(subset=["Price", "Open"])

price_vector = df["Price"].values
open_vector = df["Open"].values

r_values = np.arange(1, 11)
distances = []
for r in r_values:
    dist = distance.minkowski(price_vector, open_vector, r)
    distances.append(dist)

plt.figure(figsize=(8, 5))
plt.plot(r_values, distances, marker='s', linestyle='--', color="darkblue", markersize=6, linewidth=2)
plt.xlabel("Minkowski Parameter (r)")
plt.ylabel("Minkowski Distance")
plt.title("Minkowski Distance between Stock Price and Open Price")
plt.xticks(r_values)   
plt.grid(axis="y", linestyle="dotted", alpha=0.7)
plt.show()
