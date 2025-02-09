 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

file_path = "/content/Lab Session Data.xlsx"  # Ensure file exists
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
closing_price_col = 'Close' if 'Close' in df.columns else 'Price'
df['Trend'] = (df[closing_price_col] > df['Open']).astype(int)
X = df[['Open', 'Low']]
y = df['Trend']
X = X.apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(y, errors='coerce')
df = df.dropna(subset=['Open', 'Low', closing_price_col])   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')  
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)   
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
report = classification_report(y_test, y_pred)
print(report)
