#!pip install lime
import lime.lime_tabular
import shap
import numpy as np

def explain_with_lime(X, model, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )

    instance = X.iloc[0]
    pred = model.predict_proba([instance.values])[0]
    label_to_explain = np.argmax(pred)  # pick the predicted class

    exp = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=model.predict_proba,
        labels=(label_to_explain,)
    )
    exp.show_in_notebook()
def explain_with_shap(X, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

class_names = model.classes_.astype(str).tolist()

explain_with_lime(X_test, model, feature_names=X.columns.tolist(), class_names=class_names)
explain_with_shap(X_test, model)
