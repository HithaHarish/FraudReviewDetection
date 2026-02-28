import pandas as pd

def build_training_dataset(
    textual_df,
    reviews_df,
    customer_behavior_df,
    product_behavior_df,
    product_temporal_df,
    customer_temporal_df
):

    base = reviews_df[["review_id", "customer_id", "product_id"]]

    df = base.merge(textual_df, on="review_id", how="left")

    df = df.merge(customer_behavior_df, on="customer_id", how="left")
    df = df.merge(customer_temporal_df, on="customer_id", how="left")

    df = df.merge(product_behavior_df, on="product_id", how="left")
    df = df.merge(product_temporal_df, on="product_id", how="left")

    df = df.fillna(0)

    return df