# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# Define constants
FILE_PATH = "/content/Lab Session Data.xlsx"  # Update this path as needed
SHEET_NAME = "IRCTC Stock Price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_COLUMNS = ['Open', 'High', 'Low']
TARGET_COLUMN = 'Price'

# Function to load data
def load_data(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to preprocess data
def preprocess_data(df, feature_columns, target_column):
    if df is None:
        return None, None
    
    df = df[feature_columns + [target_column]].dropna()
    for col in feature_columns + [target_column]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    X = df[feature_columns]
    y = df[target_column]
    return X, y

# Function to train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X, y, dataset_type):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"{dataset_type} Data Metrics:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}")
    print(f"R² Score: {r2}\n")

# Main execution
if __name__ == "__main__":
    df = load_data(FILE_PATH, SHEET_NAME)
    X, y = preprocess_data(df, FEATURE_COLUMNS, TARGET_COLUMN)
    
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        model = train_model(X_train, y_train)
        
        evaluate_model(model, X_train, y_train, "Training")
        evaluate_model(model, X_test, y_test, "Test")
