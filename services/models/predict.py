import joblib
import pandas as pd


def load_model():
    saved = joblib.load("services/models/xgb_fraud_model.pkl")
    return saved   # returns dictionary


def predict_review(model, feature_list, review_row):
    X = review_row.copy()
    for col in feature_list:
        if col not in X.columns:
            X[col] = 0

    # Keep only training features and force numeric dtype expected by XGBoost.
    X = X[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    prob = model.predict_proba(X)[:, 1][0]
    prediction = model.predict(X)[0]

    return prediction, prob