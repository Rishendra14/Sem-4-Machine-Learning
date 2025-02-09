import numpy as np
import pandas as pd
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=21)

knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn.fit(X_train, y_train)

score = knn.score(X_test, y_test)
print("Model R² Score:", score)
