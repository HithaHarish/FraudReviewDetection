import joblib


def load_model():
    saved = joblib.load("services/models/xgb_fraud_model.pkl")
    return saved   # returns dictionary


def predict_review(model, feature_list, review_row):

    X = review_row.copy()

    # Keep only training features
    X = X[feature_list]

    prob = model.predict_proba(X)[:, 1][0]
    prediction = model.predict(X)[0]

    return prediction, prob