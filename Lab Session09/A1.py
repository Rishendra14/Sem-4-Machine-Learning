# A1: Stacking Regressor Implementation

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
# A1: Stacking Regressor Implementation

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load and prepare the data
from google.colab import files

uploaded = files.upload()

df=pd.read_excel("219_student.xlsx")
df["Duration"] = df["End time"] - df["Start time"]
X = df[["Start time", "End time"]]
y = df["Duration"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('knn', KNeighborsRegressor())
]

# Meta-model
meta_model = LinearRegression()

# Stacking Regressor
stack = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Stacking Regressor MSE: {mse:.2f}")
y = df["Duration"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
    ('knn', KNeighborsRegressor())
]

# Meta-model
meta_model = LinearRegression()

# Stacking Regressor
stack = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Stacking Regressor MSE: {mse:.2f}")
