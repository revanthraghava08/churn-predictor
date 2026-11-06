import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score

CSV_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

USE_N_ADDONS = False
USE_TENURE_GROUP = False
C = 0.1
THRESHOLD = 0.1

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]

def add_features(X):
    X = X.copy()
    if USE_N_ADDONS:
        X["n_addons"] = (X[ADDONS] == "Yes").sum(axis=1)
    if USE_TENURE_GROUP:
        X["tenure_group"] = pd.cut(
            X["tenure"], bins=[-1, 6, 12, 24, 48, 100],
            labels=["0-6", "7-12", "13-24", "25-48", "49+"]
        ).astype(str)
    return X

def load_data(path):
    df = pd.read_csv(path).drop(columns="customerID")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = (df["Churn"] == "Yes").astype(int)
    X = add_features(df.drop(columns="Churn"))
    return X, y

def build_model(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return make_pipeline(prep, LogisticRegression(C=C, max_iter=2000))

if __name__ == "__main__":
    X, y = load_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X_train).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    print("FINAL TEST (run once):")
    print("auc      :", round(roc_auc_score(y_test, proba), 3))
    print("confusion:", confusion_matrix(y_test, pred).tolist())
    print("recall   :", round(recall_score(y_test, pred), 3),
          "precision:", round(precision_score(y_test, pred), 3))

    final = build_model(X).fit(X, y)
    joblib.dump(final, "model.joblib")
    meta = {"threshold": THRESHOLD, "input_columns": X.columns.tolist(),
            "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__}
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved model.joblib and model_meta.json")