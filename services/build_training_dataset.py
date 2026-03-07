import pandas as pd

def build_training_dataset(
textual_df,
reviews_df,
customer_behavior_df,
product_behavior_df,
product_temporal_df,
customer_temporal_df
):

    # --------------------------------------------------
    # Merge textual features with base review data
    # --------------------------------------------------
    df = reviews_df.merge(
        textual_df,
        on="review_id",
        how="left"
    )

    # --------------------------------------------------
    # IMPORTANT: Verified Purchase Feature
    # --------------------------------------------------

    df["verified_purchase"] = df["verified_purchase"].astype(int)

    df["unverified_purchase_flag"] = (
        df["verified_purchase"] == 0
    ).astype(int)

    # --------------------------------------------------
    # Merge customer behavioral features
    # --------------------------------------------------

    df = df.merge(
        customer_behavior_df,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------
    # Merge product behavioral features
    # --------------------------------------------------

    df = df.merge(
        product_behavior_df,
        on="product_id",
        how="left"
    )

    # --------------------------------------------------
    # Merge product temporal features
    # --------------------------------------------------

    df = df.merge(
        product_temporal_df,
        on="product_id",
        how="left"
    )

    # --------------------------------------------------
    # Merge customer temporal features
    # --------------------------------------------------

    df = df.merge(
        customer_temporal_df,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------
    # Fill missing values
    # --------------------------------------------------

    df = df.fillna(0)

    return df

