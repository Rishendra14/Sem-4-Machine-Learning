from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import LabelEncoder

def apply_sfs(X, y):
    # Create a LabelEncoder object
    encoder = LabelEncoder()

    # Select object columns and apply label encoding
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = encoder.fit_transform(X[col])

    model = RandomForestClassifier(random_state=42)
    sfs = SequentialFeatureSelector(model, n_features_to_select="auto", direction='forward', scoring='accuracy', cv=5)
    sfs.fit(X, y)
    return sfs.transform(X)

# Main
X_sfs = apply_sfs(X, y)
acc_sfs = train_model(X_sfs, y)
print(f"Accuracy with Sequential Feature Selection: {acc_sfs:.2f}")