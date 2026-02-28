import pandas as pd
import numpy as np

def product_behavioral_features(reviews_df):

    reviews_df["review_timestamp"] = pd.to_datetime(reviews_df["review_timestamp"])

    product_group = reviews_df.groupby("product_id")

    product_df = pd.DataFrame()

    # --------------------------------------------------
    # Basic Metrics
    # --------------------------------------------------
    product_df["total_reviews"] = product_group["review_id"].count()
    product_df["avg_rating"] = product_group["rating"].mean()

    # --------------------------------------------------
    # Verified Review Ratio
    # --------------------------------------------------
    product_df["verified_review_ratio"] = product_group["verified_purchase"].mean()

    # --------------------------------------------------
    # Refund vs Rating Mismatch
    # --------------------------------------------------
    def refund_rating_mismatch(x):
        return ((x["rating"] >= 4) & (x["refunded_product"] == True)).mean()

    product_df["refund_rating_mismatch_ratio"] = (
        reviews_df.groupby("product_id")
        .apply(refund_rating_mismatch)
    )

    # --------------------------------------------------
    # Same Customer Cross-Platform Ratio
    # --------------------------------------------------
    multi_platform = (
        reviews_df.groupby(["product_id", "customer_id"])["platform_id"]
        .nunique()
        .reset_index()
    )

    cross_platform = (
        multi_platform[multi_platform["platform_id"] > 1]
        .groupby("product_id")
        .size()
    )

    unique_customers = reviews_df.groupby("product_id")["customer_id"].nunique()

    product_df["cross_platform_customer_ratio"] = (
        cross_platform / unique_customers
    )

    # --------------------------------------------------
    # Same Platform Repeat Ratio
    # --------------------------------------------------
    same_platform_repeat = (
        reviews_df.groupby(
            ["product_id", "customer_id", "platform_id"]
        )["review_id"]
        .count()
        .reset_index()
    )

    repeat_counts = (
        same_platform_repeat[same_platform_repeat["review_id"] > 1]
        .groupby("product_id")
        .size()
    )

    product_df["same_platform_repeat_ratio"] = (
        repeat_counts / unique_customers
    )

    # --------------------------------------------------
    # Review Spike Count (Hourly Burst)
    # --------------------------------------------------
    reviews_sorted = reviews_df.sort_values(["product_id", "review_timestamp"])

    reviews_sorted["reviews_last_hour"] = (
        reviews_sorted
        .groupby("product_id")
        .rolling("1H", on="review_timestamp")
        .count()["review_id"]
        .reset_index(level=0, drop=True)
    )

    spike_counts = (
        reviews_sorted[reviews_sorted["reviews_last_hour"] > 10]
        .groupby("product_id")
        .size()
    )

    product_df["review_spike_count"] = spike_counts

    # --------------------------------------------------
    # Unique Review Ratio
    # --------------------------------------------------
    unique_review_ratio = (
        reviews_df.groupby("product_id")["review_text"]
        .nunique()
        / product_df["total_reviews"]
    )

    product_df["unique_review_ratio"] = unique_review_ratio

    # --------------------------------------------------
    # Clean Missing
    # --------------------------------------------------
    # Convert index (product_id) back to column
    product_df = product_df.reset_index()

    # Now select only needed columns
    product_df = product_df[
        [
            "product_id",
            "total_reviews",
            "avg_rating",
            "verified_review_ratio",
            "refund_rating_mismatch_ratio",
            "cross_platform_customer_ratio",
            "same_platform_repeat_ratio",
            "review_spike_count",
            "unique_review_ratio"
        ]
    ]

    return product_df