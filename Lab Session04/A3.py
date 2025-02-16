# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define constants
FILE_PATH = "/content/Lab Session Data.xlsx"  # Update this path as needed
SHEET_NAME = "IRCTC Stock Price"
NUM_POINTS = 20
FEATURE_COLUMNS = ['Open', 'Price']

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
def preprocess_data(df, feature_columns, num_points):
    if df is None:
        return None, None, None
    
    df = df[feature_columns].dropna()
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    X = df['Open'].values[:num_points]
    Y = df['Price'].values[:num_points]
    classes = np.where(Y < X, 0, 1)
    
    return X, Y, classes

# Function to plot data
def plot_data(X, Y, classes):
    plt.figure(figsize=(8, 6))
    for i in range(len(X)):
        color = 'blue' if classes[i] == 0 else 'red'
        label = 'Class 0 (Down - Blue)' if classes[i] == 0 and i == 0 else ('Class 1 (Up - Red)' if classes[i] == 1 and i == 0 else "")
        plt.scatter(X[i], Y[i], color=color, label=label)
    
    plt.title("IRCTC Stock Price Movement (Class 0 - Blue, Class 1 - Red)")
    plt.xlabel("Open Price")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    df = load_data(FILE_PATH, SHEET_NAME)
    X, Y, classes = preprocess_data(df, FEATURE_COLUMNS, NUM_POINTS)
    
    if X is not None and Y is not None and classes is not None:
        plot_data(X, Y, classes)
