from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float).fillna(series.median())
    if s.nunique() <= 1:
        return pd.Series(0.0, index=series.index)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return pd.Series(
        scaler.fit_transform(s.to_numpy().reshape(-1, 1)).reshape(-1),
        index=series.index,
    )


def _compute_burst_density(
    df: pd.DataFrame,
    date_col: str,
    product_col: str = "Product_ID",
    window_days: int = 3,
) -> pd.Series:
    """
    For each review, count how many reviews for the same product occur
    within a +/- window_days interval around its date.
    Higher density indicates more temporal suspicion.
    """
    if date_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    density = pd.Series(0, index=df.index, dtype=float)

    grouped = df.assign(_date=dates).groupby(product_col)
    for _, grp in grouped:
        grp = grp.sort_values("_date")
        times = grp["_date"]
        for idx, current_time in times.items():
            if pd.isna(current_time):
                density.loc[idx] = 0.0
                continue
            window_start = current_time - timedelta(days=window_days)
            window_end = current_time + timedelta(days=window_days)
            mask = (times >= window_start) & (times <= window_end)
            density.loc[idx] = float(mask.sum())

    return _normalize(density)


def compute_temporal_score(
    df: pd.DataFrame,
    date_col: str = "Review_Date",
    product_col: str = "Product_ID",
) -> pd.Series:
    """
    Compute temporal fraud score (0–100) using:
        - Reviews_Per_Day (higher => more suspicious)
        - Burst_Flag (1 => more suspicious)
        - Local product-level review density around the review date
    """
    df = df.copy()

    components = []

    if "Reviews_Per_Day" in df.columns:
        components.append(_normalize(df["Reviews_Per_Day"]))

    if "Burst_Flag" in df.columns:
        components.append(_normalize(df["Burst_Flag"]))

    density = _compute_burst_density(df, date_col=date_col, product_col=product_col)
    components.append(density)

    if not components:
        return pd.Series(0.0, index=df.index, name="Temporal_Score_Final")

    stacked = np.vstack([c.to_numpy() for c in components])
    mean_risk = stacked.mean(axis=0)

    scaler = MinMaxScaler(feature_range=(0, 100))
    scores = scaler.fit_transform(mean_risk.reshape(-1, 1)).reshape(-1)
    return pd.Series(scores, index=df.index, name="Temporal_Score_Final")


__all__ = ["compute_temporal_score"]
