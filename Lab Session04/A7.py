import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define file path
file_path = "/content/Lab Session Data.xlsx"  # Update this path as needed

def load_data(file_path, sheet_name='IRCTC Stock Price'):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df[['Open', 'Price']].dropna()
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def prepare_data(df, sample_size=100):
    X = df[['Open', 'Price']].values[:sample_size]
    y = np.where(X[:, 1] < X[:, 0], 0, 1)
    return train_test_split(X, y, test_size=0.3, random_state=42)

def find_best_k(X_train, y_train):
    param_grid = {'n_neighbors': np.arange(1, 21)}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_['n_neighbors'], grid_search.best_score_, grid_search

def train_and_evaluate_knn(X_train, X_test, y_train, y_test, best_k):
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix

def plot_accuracy(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results['param_n_neighbors'], results['mean_test_score'], marker='o', linestyle='-')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title("k-NN Accuracy for Different k Values")
    plt.grid(True)
    plt.show()

def main():
    df = load_data(file_path)
    if df is not None:
        X_train, X_test, y_train, y_test = prepare_data(df)
        best_k, best_score, grid_search = find_best_k(X_train, y_train)
        print(f"Best k value: {best_k}")
        print(f"Best cross-validation accuracy: {best_score:.4f}")
        
        test_accuracy, report, matrix = train_and_evaluate_knn(X_train, X_test, y_train, y_test, best_k)
        print("\nTest Accuracy:", test_accuracy)
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(matrix)
        
        plot_accuracy(pd.DataFrame(grid_search.cv_results_))

if __name__ == "__main__":
    main()