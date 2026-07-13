import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

X_train = pd.read_csv("artifacts/X_train.csv")
X_test = pd.read_csv("artifacts/X_test.csv")

y_train = pd.read_csv("artifacts/y_train.csv").squeeze()
y_test = pd.read_csv("artifacts/y_test.csv").squeeze()

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    results_df = pd.DataFrame(results)

print(results_df)

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print(best_model)

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

best_model_object = models[best_model_name]

joblib.dump(
    best_model_object,
    "models/best_model.pkl"
)

print(f"Best model saved: {best_model_name}")

