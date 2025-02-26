# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Define file path and sheet name
FILE_PATH = "/content/Lab Session Data.xlsx"  # Update path as necessary
SHEET_NAME = "IRCTC Stock Price"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Price']
TARGET_COLUMN = 'Target'

# Function to load Excel data
def load_data(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to preprocess the dataset
def preprocess_data(df):
    if df is None or df.empty:
        print("Dataframe is empty or None. Skipping preprocessing.")
        return None
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(inplace=True)
    df['Next_Day_Price'] = df['Price'].shift(-1)
    df['Target'] = (df['Next_Day_Price'] > df['Price']).astype(int)
    df.dropna(inplace=True)
    return df

# Function to split and standardize data
def split_and_scale_data(df, feature_columns, target_column, test_size, random_state):
    if df is None:
        print("No valid data available for splitting and scaling.")
        return None, None, None, None
    
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Function to train the random forest model
def train_random_forest(X_train, y_train, random_state):
    if X_train is None or y_train is None:
        print("Training data is missing. Cannot train model.")
        return None
    
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X, y, dataset_type):
    if model is None or X is None or y is None:
        print(f"Skipping evaluation for {dataset_type} data due to missing inputs.")
        return None, None
    
    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    
    print(f"\nConfusion Matrix - {dataset_type} Data:")
    print(conf_matrix)
    print(f"\nClassification Report - {dataset_type} Data:")
    print(report)
    
    return conf_matrix, report

# Main execution
if __name__ == "__main__":
    df = load_data(FILE_PATH, SHEET_NAME)
    df = preprocess_data(df)
    
    if df is not None:
        X_train, X_test, y_train, y_test = split_and_scale_data(df, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
        model = train_random_forest(X_train, y_train, RANDOM_STATE)
        
        evaluate_model(model, X_train, y_train, "Training")
        evaluate_model(model, X_test, y_test, "Test")
    else:
        print("No valid data to proceed further.")