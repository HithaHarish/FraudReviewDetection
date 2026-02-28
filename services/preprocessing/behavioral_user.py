import pandas as pd
from datetime import datetime

def customer_behavioral_features(users_df, reviews_df):

    today = pd.Timestamp.today()

    # Ensure datetime
    users_df["account_created"] = pd.to_datetime(users_df["account_created"])
    reviews_df["review_timestamp"] = pd.to_datetime(reviews_df["review_timestamp"])

    # ---------------------------------------
    # Base Aggregation
    # ---------------------------------------
    user_group = reviews_df.groupby("customer_id")

    behavioral_df = pd.DataFrame()

    behavioral_df["total_reviews"] = user_group["review_id"].count()
    behavioral_df["avg_rating_by_user"] = user_group["rating"].mean()
    behavioral_df["positive_review_ratio"] = user_group["rating"].apply(
        lambda x: (x >= 4).mean()
    )
    behavioral_df["verified_review_ratio"] = user_group["verified_purchase"].mean()
    behavioral_df["refund_ratio"] = user_group["refunded_product"].mean()

    # ---------------------------------------
    # Account Age
    # ---------------------------------------
    users_df = users_df.set_index("customer_id")

    behavioral_df = behavioral_df.merge(
        users_df[["account_created"]],
        left_index=True,
        right_index=True,
        how="left"
    )

    behavioral_df["account_age_days"] = (
        today - behavioral_df["account_created"]
    ).dt.days

    behavioral_df["account_age_weeks"] = behavioral_df["account_age_days"] / 7

    # ---------------------------------------
    # Review Frequency
    # ---------------------------------------
    behavioral_df["review_frequency_per_week"] = (
        behavioral_df["total_reviews"] /
        (behavioral_df["account_age_weeks"] + 1)
    )

    # ---------------------------------------
    # Same Product Reviewed on Different Platforms
    # ---------------------------------------
    multi_platform_reviews = (
        reviews_df.groupby(["customer_id", "product_id"])["platform_id"]
        .nunique()
        .reset_index()
    )

    multi_platform_flag = (
        multi_platform_reviews[multi_platform_reviews["platform_id"] > 1]
        .groupby("customer_id")
        .size()
    )

    behavioral_df["cross_platform_same_product_count"] = (
        multi_platform_flag
    )

    # ---------------------------------------
    # Same Product Reviewed Multiple Times on Same Platform
    # ---------------------------------------
    same_platform_duplicates = (
        reviews_df.groupby(
            ["customer_id", "product_id", "platform_id"]
        )["review_id"]
        .count()
        .reset_index()
    )

    same_platform_duplicates = (
        same_platform_duplicates[
            same_platform_duplicates["review_id"] > 1
        ]
        .groupby("customer_id")
        .size()
    )

    behavioral_df["same_platform_repeat_count"] = same_platform_duplicates

    # ---------------------------------------
    # Fill NaNs
    # ---------------------------------------

    # Convert index (customer_id) back to column
    behavioral_df = behavioral_df.reset_index()

    # Select only required columns
    behavioral_df = behavioral_df[
        [
            "customer_id",
            "total_reviews",
            "avg_rating_by_user",
            "positive_review_ratio",
            "verified_review_ratio",
            "refund_ratio",
            "account_age_days",
            "review_frequency_per_week",
            "cross_platform_same_product_count",
            "same_platform_repeat_count"
        ]
    ].rename(columns={
        "avg_rating_by_user": "avg_rating"
    })

    return behavioral_df