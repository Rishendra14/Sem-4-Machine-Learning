import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files

uploaded = files.upload()

df=pd.read_excel("219_student.xlsx")

def plot_correlation_heatmap(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
plot_correlation_heatmap(df)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def apply_pca(X, variance_threshold=0.99):
    # Selecting only numeric features for scaling
    numeric_features = X.select_dtypes(include=['number']).columns
    X_numeric = X[numeric_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Main
X = df.drop(columns=['Member', 'Number'])  # Dropping 'Number' column
y = df['Member']


X_pca_99 = apply_pca(X, 0.99)
acc_99 = train_model(X_pca_99, y)
print(f"Accuracy with PCA (99% variance): {acc_99:.2f}")

#A3