 
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

file_path = "/content/Lab Session Data.xlsx"   
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
price_bins = [0, 2000, 2500, np.inf]   
price_labels = ["Low", "Mid", "High"]   
df["Price Range"] = pd.cut(df["Price"], bins=price_bins, labels=price_labels)

 
features = ["Price", "Open", "High", "Low", "Volume", "Chg%"]

df["Volume"] = df["Volume"].astype(str).replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval)

class_stats = df.groupby("Price Range")[features].agg(['mean', 'std'])
class_centroids = class_stats.xs('mean', axis=1, level=1).loc[["Low", "Mid"]]
class_spreads = class_stats.xs('std', axis=1, level=1).loc[["Low", "Mid"]]
distance_matrix = squareform(pdist(class_centroids, metric="euclidean"))
distance_df = pd.DataFrame(distance_matrix, index=class_centroids.index, columns=class_centroids.index)

print("Class Centroids (Mean Values)")
print(class_centroids)

print("Intraclass Spread (Standard Deviation) ")
print(class_spreads)

print(" Interclass Distances")
for i in range(len(class_centroids.index)):
    for j in range(i + 1, len(class_centroids.index)):
        print(f"Distance between {class_centroids.index[i]} and {class_centroids.index[j]}: {distance_matrix[i][j]:.2f}")
