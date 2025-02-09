import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   

file_path = "/content/Lab Session Data.xlsx"  
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")


df.dropna(subset=["Price"], inplace=True)
bins = np.linspace(df["Price"].min(), df["Price"].max(), num=10)
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=bins, kde=True, color="royalblue", edgecolor="black", alpha=0.75)
plt.xlabel("Stock Price", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Distribution of IRCTC Stock Prices", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

mean_price = np.mean(df["Price"])
variance_price = np.var(df["Price"], ddof=1)  

print(f"Mean Stock Price: {mean_price:.2f}")
print(f"Variance: {variance_price:.2f}")
