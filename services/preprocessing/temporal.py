import pandas as pd
import numpy as np

def temporal_features(reviews_df):

    reviews_df["review_timestamp"] = pd.to_datetime(reviews_df["review_timestamp"])
    reviews_df = reviews_df.sort_values("review_timestamp")

    # --------------------------------------------------
    # PRODUCT LEVEL TEMPORAL FEATURES
    # --------------------------------------------------
    product_temporal = pd.DataFrame()

    product_group = reviews_df.groupby("product_id")

    # Total reviews
    product_temporal["total_reviews"] = product_group["review_id"].count()

    # --------------------------------------------------
    # Short Time Gaps (Product)
    # --------------------------------------------------
    def avg_time_gap(x):
        x = x.sort_values("review_timestamp")
        gaps = x["review_timestamp"].diff().dt.total_seconds()
        return gaps.mean()

    product_temporal["avg_time_gap_seconds"] = product_group.apply(avg_time_gap)

    # --------------------------------------------------
    # Review Burst Count (1 Hour Window)
    # --------------------------------------------------
    reviews_df["hour_window"] = reviews_df["review_timestamp"].dt.floor("H")

    burst_counts = (
        reviews_df.groupby(["product_id", "hour_window"])
        .size()
        .reset_index(name="count")
    )

    burst_flag = (
        burst_counts[burst_counts["count"] > 10]
        .groupby("product_id")
        .size()
    )

    product_temporal["review_burst_count"] = burst_flag

    # --------------------------------------------------
    # Coordinated Multi-User Activity (5 min window)
    # --------------------------------------------------
    reviews_df["five_min_window"] = reviews_df["review_timestamp"].dt.floor("5T")

    coordinated = (
        reviews_df.groupby(["product_id", "five_min_window"])["customer_id"]
        .nunique()
        .reset_index(name="unique_users")
    )

    coordinated_flag = (
        coordinated[coordinated["unique_users"] > 5]
        .groupby("product_id")
        .size()
    )

    product_temporal["coordinated_multi_user_events"] = coordinated_flag

    # --------------------------------------------------
    # Unnatural Posting Hours (Product)
    # --------------------------------------------------
    reviews_df["hour"] = reviews_df["review_timestamp"].dt.hour

    def night_ratio(x):
        return ((x["hour"] >= 1) & (x["hour"] <= 4)).mean()

    product_temporal["night_posting_ratio"] = (
        reviews_df.groupby("product_id").apply(night_ratio)
    )

    # --------------------------------------------------
    # Periodicity (Low gap variance)
    # --------------------------------------------------
    def gap_variance(x):
        x = x.sort_values("review_timestamp")
        gaps = x["review_timestamp"].diff().dt.total_seconds()
        return gaps.var()

    product_temporal["gap_variance"] = product_group.apply(gap_variance)

    # --------------------------------------------------
    # Sudden Surge Detection (Last 7 Days)
    # --------------------------------------------------
    latest_date = reviews_df["review_timestamp"].max()
    last_week = latest_date - pd.Timedelta(days=7)

    last_week_counts = (
        reviews_df[reviews_df["review_timestamp"] >= last_week]
        .groupby("product_id")
        .size()
    )

    historical_avg = (
        product_temporal["total_reviews"] / 
        ((latest_date - reviews_df["review_timestamp"].min()).days / 7)
    )

    product_temporal["sudden_surge_ratio"] = (
        last_week_counts / historical_avg
    )

    product_temporal = product_temporal.fillna(0).reset_index()

    # --------------------------------------------------
    # CUSTOMER LEVEL TEMPORAL FEATURES
    # --------------------------------------------------
    customer_temporal = pd.DataFrame()

    customer_group = reviews_df.groupby("customer_id")

    # Customer burst activity
    reviews_df["customer_hour_window"] = reviews_df["review_timestamp"].dt.floor("H")

    customer_burst = (
        reviews_df.groupby(["customer_id", "customer_hour_window"])
        .size()
        .reset_index(name="count")
    )

    customer_burst_flag = (
        customer_burst[customer_burst["count"] > 5]
        .groupby("customer_id")
        .size()
    )

    customer_temporal["customer_burst_count"] = customer_burst_flag

    # Unnatural posting hours (Customer)
    customer_temporal["customer_night_ratio"] = (
        reviews_df.groupby("customer_id").apply(night_ratio)
    )

    customer_temporal = customer_temporal.fillna(0).reset_index()

    product_temporal = product_temporal[
        [
            "product_id",
            "avg_time_gap_seconds",
            "review_burst_count",
            "coordinated_multi_user_events",
            "night_posting_ratio",
            "gap_variance",
            "sudden_surge_ratio"
        ]
    ]

    customer_temporal = customer_temporal[
        [
            "customer_id",
            "customer_burst_count",
            "customer_night_ratio"
        ]
    ]

    return product_temporal, customer_temporal