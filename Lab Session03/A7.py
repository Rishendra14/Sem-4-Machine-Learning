 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

 
file_path = "/content/Lab Session Data.xlsx"   
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

X = df[['Low', 'Open']]   
y = df['High']
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
df = df.dropna(subset=['Low', 'Open', 'High'])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=24)
k_values = range(1, 13)   
scores = []
for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')   
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores.append(score)
    print(f"k={k}, R² Score: {score:.10f}")   
 
plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker='o', linestyle='dashed', color='b', label="R² Score")
plt.xlabel('k Value')
plt.ylabel('R² Score')
plt.title('kNN Accuracy for Different k Values')
plt.xticks(k_values)
plt.grid(True)
best_k = k_values[np.argmax(scores)]
best_score = max(scores)
plt.annotate(f"Best k={best_k}", xy=(best_k, best_score), xytext=(best_k + 1, best_score - 0.002),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, color='red')

plt.legend()
plt.show()
