import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/customer_churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

X = df.drop("Churn", axis=1)
y = df["Churn"]

le = LabelEncoder()
y = le.fit_transform(y)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

os.makedirs("artifacts", exist_ok=True)

X_train.to_csv("artifacts/X_train.csv", index=False)
X_test.to_csv("artifacts/X_test.csv", index=False)

pd.DataFrame(y_train).to_csv("artifacts/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("artifacts/y_test.csv", index=False)

print("Processed data saved successfully!")

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)