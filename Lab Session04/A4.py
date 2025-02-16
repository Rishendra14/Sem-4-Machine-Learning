# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define constants
FILE_PATH = "/content/Lab Session Data.xlsx"  # Update this path as needed
SHEET_NAME = "IRCTC Stock Price"
NUM_TRAIN_POINTS = 20
K_NEIGHBORS = 3
TEST_RANGE = np.arange(0, 10.1, 0.1)

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
def preprocess_data(df, num_train_points):
    if df is None:
        return None, None
    
    df = df[['Open', 'Price']].dropna()
    for col in ['Open', 'Price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    X_train = df[['Open', 'Price']].values[:num_train_points]
    y_train = np.where(X_train[:, 1] < X_train[:, 0], 0, 1)
    
    return X_train, y_train

# Function to train k-NN classifier
def train_knn_classifier(X_train, y_train, k_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_train, y_train)
    return knn

# Function to generate test data
def generate_test_data(test_range):
    X_test, Y_test = np.meshgrid(test_range, test_range)
    test_points = np.c_[X_test.ravel(), Y_test.ravel()]
    return X_test, Y_test, test_points

# Function to plot classification results
def plot_classification(X_test, Y_test, predicted_classes, X_train, y_train):
    plt.figure(figsize=(10, 8))
    plt.contourf(X_test, Y_test, predicted_classes, alpha=0.5, cmap='bwr')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0 (Train - Blue)')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1 (Train - Red)')
    
    plt.title("k-NN Classification (k=3) - IRCTC Stock Price")
    plt.xlabel("Open Price")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    df = load_data(FILE_PATH, SHEET_NAME)
    X_train, y_train = preprocess_data(df, NUM_TRAIN_POINTS)
    
    if X_train is not None and y_train is not None:
        knn = train_knn_classifier(X_train, y_train, K_NEIGHBORS)
        X_test, Y_test, test_points = generate_test_data(TEST_RANGE)
        predicted_classes = knn.predict(test_points).reshape(X_test.shape)
        plot_classification(X_test, Y_test, predicted_classes, X_train, y_train)
