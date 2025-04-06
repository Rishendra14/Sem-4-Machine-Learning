# A2: Pipeline with Preprocessing and Stacking

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create pipeline with scaler and stacking model
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('stack', stack)
])

# Fit and predict
pipeline.fit(X_train, y_train)
pipeline_pred = pipeline.predict(X_test)

# Evaluate
pipeline_mse = mean_squared_error(y_test, pipeline_pred)
print(f"Pipeline MSE: {pipeline_mse:.2f}")
