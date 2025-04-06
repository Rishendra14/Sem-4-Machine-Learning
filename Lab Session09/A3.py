# A3: LIME Explanation
!pip install lime
import lime
import lime.lime_tabular
import numpy as np

# LIME explainer setup
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    mode='regression'
)

# Explain a single prediction
sample_idx = 0
exp = explainer.explain_instance(X_test.values[sample_idx], pipeline.predict, num_features=2)
exp.show_in_notebook()
