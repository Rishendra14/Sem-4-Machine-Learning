# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define the file path
FILE_PATH = "/content/Lab Session Data.xlsx"  # Update this path as needed
SHEET_NAME = 'IRCTC Stock Price'
K_VALUES = [1, 3, 5, 7, 9]

# Function to load data
def load_data(file_path, sheet_name):
    try:
        df = pd.ExcelFile(file_path)
        df_irctc = pd.read_excel(df, sheet_name=sheet_name)
        return df_irctc[['Open', 'Price']].dropna()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to preprocess data
def preprocess_data(df):
    if df is None:
        return None
    for col in ['Open', 'Price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

# Function to prepare training data
def prepare_training_data(df):
    X_train = df[['Open', 'Price']].values[:20]
    y_train = np.where(X_train[:, 1] < X_train[:, 0], 0, 1)
    return X_train, y_train

# Function to generate test data
def generate_test_data():
    x_test_values = np.arange(0, 10.1, 0.1)
    y_test_values = np.arange(0, 10.1, 0.1)
    X_test, Y_test = np.meshgrid(x_test_values, y_test_values)
    return np.c_[X_test.ravel(), Y_test.ravel()], X_test, Y_test

# Function to train and plot k-NN classifiers
def train_and_plot_knn(X_train, y_train, test_points, X_test, Y_test, k_values):
    plt.figure(figsize=(20, 15))
    for idx, k in enumerate(k_values):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predicted_classes = knn.predict(test_points).reshape(X_test.shape)

        plt.subplot(2, 3, idx + 1)
        plt.contourf(X_test, Y_test, predicted_classes, alpha=0.5, cmap='bwr')
        plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Class 0 (Train - Blue)')
        plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Class 1 (Train - Red)')
        plt.title(f"k-NN Classification (k={k})")
        plt.xlabel("Open Price")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main execution
def main():
    df = load_data(FILE_PATH, SHEET_NAME)
    df = preprocess_data(df)
    if df is not None:
        X_train, y_train = prepare_training_data(df)
        test_points, X_test, Y_test = generate_test_data()
        train_and_plot_knn(X_train, y_train, test_points, X_test, Y_test, K_VALUES)

if __name__ == "__main__":
    main()
