import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def train_xgboost_model(training_df):
    # -------------------------------------------------
    # Remove non-feature columns
    # -------------------------------------------------

    drop_columns = [
        "review_id",
        "customer_id",
        "product_id",
        "review_text",
        "review_timestamp",
        "account_created",
        "label"
    ]

    existing_cols = [c for c in drop_columns if c in training_df.columns]

    X = training_df.drop(columns=existing_cols, errors="ignore")

    y = training_df["label"]

    # -------------------------------------------------
    # Train-Test Split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------
    # Handle Class Imbalance
    # -------------------------------------------------

    fraud_count = sum(y)
    non_fraud_count = len(y) - fraud_count

    scale_pos_weight = non_fraud_count / (fraud_count + 1)

    # -------------------------------------------------
    # XGBoost Model
    # -------------------------------------------------

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )

    model.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # -------------------------------------------------
    # Save Model
    # -------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "features": X.columns.tolist()
        },
        "services/models/xgb_fraud_model.pkl"
    )

    return model, accuracy, roc_auc

