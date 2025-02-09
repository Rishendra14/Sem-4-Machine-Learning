import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print("Model R² Score:", score)
y_pred = knn.predict(X_test)
print("Predicted values:", y_pred)
test_vect = X_test[0].reshape(1, -1)   
predicted_value = knn.predict(test_vect)
print("Prediction for first test vector:", predicted_value)
 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
