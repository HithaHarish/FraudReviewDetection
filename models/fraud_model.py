import numpy as np
import pandas as pd

from utils.textual_layer import compute_textual_score
from utils.behavioral_layer import compute_behavioral_score
from utils.temporal_layer import compute_temporal_score


RANDOM_STATE = 42


def load_and_integrate_data(
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Load customer, product and review data, then merge into a single DataFrame.

    The function is tolerant to the exact filenames used in the prompt and will
    try multiple options.
    """
    # Customer data
    customer_paths = [
        "customers.csv",
        "customer_preprocessed_only_200_rows.csv",
    ]
    product_paths = [
        "products.csv",
        "product_preprocessed_only_200_rows.csv",
    ]
    review_paths = [
        "reviews.xlsx",
        "textual_temporal_200_rows.csv.xlsx",
    ]

    def _load_first_ok(paths, reader):
        last_err = None
        for p in paths:
            try:
                return reader(f"{data_dir}/{p}")
            except Exception as e:  # noqa: BLE001
                last_err = e
        if last_err is not None:
            raise last_err
        raise FileNotFoundError(f"None of {paths} could be loaded from {data_dir}")

    customers = _load_first_ok(customer_paths, pd.read_csv)
    products = _load_first_ok(product_paths, pd.read_csv)
    reviews = _load_first_ok(review_paths, pd.read_excel)

    # Basic normalization of column names
    for df in (customers, products, reviews):
        df.columns = [str(c).strip() for c in df.columns]

    # Ensure key columns exist / alias if needed
    def _ensure_col(df, candidates, required=True):
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise KeyError(f"None of the columns {candidates} found in {list(df.columns)}")
        return None

    customer_id_col_customers = _ensure_col(customers, ["Customer_ID", "customer_id", "CustomerId"])
    customer_id_col_reviews = _ensure_col(reviews, ["Customer_ID", "customer_id", "CustomerId"])
    product_id_col_products = _ensure_col(products, ["Product_ID", "product_id", "ProductId"])
    product_id_col_reviews = _ensure_col(reviews, ["Product_ID", "product_id", "ProductId"])

    # Merge customers -> reviews
    merged = reviews.merge(
        customers,
        left_on=customer_id_col_reviews,
        right_on=customer_id_col_customers,
        how="left",
        suffixes=("", "_cust"),
    )

    # Merge products
    merged = merged.merge(
        products,
        left_on=product_id_col_reviews,
        right_on=product_id_col_products,
        how="left",
        suffixes=("", "_prod"),
    )

    # Handle missing numeric values with simple imputation (median)
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    # Simple normalization for numeric features we know are used as inputs
    # Each layer will still guard against missing columns.
    def _normalize_column(df: pd.DataFrame, col: str) -> None:
        if col not in df.columns:
            return
        series = df[col].astype(float)
        col_min, col_max = series.min(), series.max()
        if col_max == col_min:
            df[col + "_norm"] = 0.0
        else:
            df[col + "_norm"] = (series - col_min) / (col_max - col_min)

    for col in [
        "Account_Age",
        "Review_Frequency",
        "Refund_Ratio",
        "Verified_Purchase_Ratio",
        "Average_Rating_By_User",
        "Reviews_Per_Day",
    ]:
        _normalize_column(merged, col)

    return merged


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Textual, Behavioral, Temporal and final fraud risk scores.
    """
    df = df.copy()

    df["Text_Score"] = compute_textual_score(df, random_state=RANDOM_STATE)
    df["Behavioral_Score_Final"] = compute_behavioral_score(df)
    df["Temporal_Score_Final"] = compute_temporal_score(df)

    df["Final_Fraud_Risk_Score"] = (
        0.50 * df["Text_Score"]
        + 0.30 * df["Behavioral_Score_Final"]
        + 0.20 * df["Temporal_Score_Final"]
    )
    df["Final_Fraud_Risk_Score"] = df["Final_Fraud_Risk_Score"].clip(0, 100)

    # Risk level classification
    def _classify(score: float) -> str:
        if score >= 70:
            return "High Risk"
        if 40 <= score < 70:
            return "Moderate Risk"
        return "Low Risk"

    df["Risk_Level"] = df["Final_Fraud_Risk_Score"].apply(_classify)

    return df


def aggregate_product_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute product-level aggregation:
    - Average fraud score per Product_ID
    - Suspicious review ratio (High Risk share)
    - Product authenticity score (100 - average fraud)
    """
    product_id_col = None
    for c in ["Product_ID", "product_id", "ProductId"]:
        if c in df.columns:
            product_id_col = c
            break
    if product_id_col is None:
        raise KeyError("Product_ID column not found for aggregation.")

    grouped = df.groupby(product_id_col).agg(
        Avg_Fraud_Score=("Final_Fraud_Risk_Score", "mean"),
        Review_Count=("Final_Fraud_Risk_Score", "count"),
        High_Risk_Reviews=("Risk_Level", lambda x: (x == "High Risk").sum()),
    )

    grouped["Suspicious_Review_Ratio"] = (
        grouped["High_Risk_Reviews"] / grouped["Review_Count"]
    ).fillna(0.0)
    grouped["Product_Authenticity_Score"] = 100.0 - grouped["Avg_Fraud_Score"].clip(0, 100)

    grouped = grouped.reset_index()
    return grouped


__all__ = [
    "load_and_integrate_data",
    "compute_all_scores",
    "aggregate_product_level",
]

