import xgboost as xgb
import joblib
<<<<<<< HEAD
=======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
>>>>>>> 393cadb (Updated - changes needed in model training)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

def train_xgboost_model(training_df):
    # -------------------------------------------------
    # Remove non-feature columns
    # -------------------------------------------------

<<<<<<< HEAD
    drop_columns = [
        "review_id",
        "customer_id",
        "product_id",
        "review_text",
        "review_timestamp",
        "account_created",
        "label"
    ]
=======
    leakage_features = []
>>>>>>> 393cadb (Updated - changes needed in model training)

    existing_cols = [c for c in drop_columns if c in training_df.columns]

<<<<<<< HEAD
    X = training_df.drop(columns=existing_cols, errors="ignore")

    y = training_df["label"]

    # -------------------------------------------------
    # Train-Test Split
    # -------------------------------------------------
=======
    X = training_df.drop(
        columns=["review_id", "customer_id", "product_id", "label"] + existing_leakage,
        errors="ignore"
    )

    y = training_df["label"]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.std() > 0]

    feature_list = X.columns.tolist()
>>>>>>> 393cadb (Updated - changes needed in model training)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

<<<<<<< HEAD
    # -------------------------------------------------
    # Handle Class Imbalance
    # -------------------------------------------------

    fraud_count = sum(y)
    non_fraud_count = len(y) - fraud_count

    scale_pos_weight = non_fraud_count / (fraud_count + 1)

    # -------------------------------------------------
    # XGBoost Model
    # -------------------------------------------------

=======
>>>>>>> 393cadb (Updated - changes needed in model training)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.3,  # Forces trees to use different continuous features, creating a smooth probability distribution
        eval_metric="aucpr",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

<<<<<<< HEAD
    # -------------------------------------------------
    # Evaluation
    # -------------------------------------------------

=======
>>>>>>> 393cadb (Updated - changes needed in model training)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

<<<<<<< HEAD
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
=======
    print(classification_report(y_test, y_pred))

    joblib.dump({
        "model": model,
        "features": feature_list
    }, "services/models/xgb_fraud_model.pkl")
>>>>>>> 393cadb (Updated - changes needed in model training)

