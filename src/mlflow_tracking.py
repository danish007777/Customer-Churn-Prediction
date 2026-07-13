import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
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

mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run():

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("random_state", 42)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="logistic_regression"
    )

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)