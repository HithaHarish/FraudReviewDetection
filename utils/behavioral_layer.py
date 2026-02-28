from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BEHAVIORAL_FEATURES = [
    "Account_Age",
    "Review_Frequency",
    "Refund_Ratio",
    "Verified_Purchase_Ratio",
    "Average_Rating_By_User",
]


def _normalize_series(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Normalize a numeric series to [0, 1].
    If invert=True, high values become low risk and vice versa.
    """
    s = series.astype(float).fillna(series.median())
    if s.nunique() <= 1:
        norm = pd.Series(0.0, index=series.index)
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))
        norm = pd.Series(
            scaler.fit_transform(s.to_numpy().reshape(-1, 1)).reshape(-1),
            index=series.index,
        )
    return 1.0 - norm if invert else norm


def compute_behavioral_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute behavioral fraud score (0–100) using:
        - Account_Age (older accounts => lower risk, inverted)
        - Review_Frequency (more frequent => higher risk)
        - Refund_Ratio (higher => higher risk)
        - Verified_Purchase_Ratio (higher => lower risk, inverted)
        - Average_Rating_By_User (extremes / higher rating treated as slightly more risky)

    Missing columns are ignored gracefully.
    """
    df = df.copy()
    risk_components: List[pd.Series] = []

    if "Account_Age" in df.columns:
        risk_components.append(_normalize_series(df["Account_Age"], invert=True))
    if "Review_Frequency" in df.columns:
        risk_components.append(_normalize_series(df["Review_Frequency"], invert=False))
    if "Refund_Ratio" in df.columns:
        risk_components.append(_normalize_series(df["Refund_Ratio"], invert=False))
    if "Verified_Purchase_Ratio" in df.columns:
        risk_components.append(_normalize_series(df["Verified_Purchase_Ratio"], invert=True))
    if "Average_Rating_By_User" in df.columns:
        risk_components.append(_normalize_series(df["Average_Rating_By_User"], invert=False))

    if not risk_components:
        return pd.Series(0.0, index=df.index, name="Behavioral_Score_Final")

    stacked = np.vstack([s.to_numpy() for s in risk_components])
    mean_risk = stacked.mean(axis=0)

    scaler = MinMaxScaler(feature_range=(0, 100))
    scores = scaler.fit_transform(mean_risk.reshape(-1, 1)).reshape(-1)
    return pd.Series(scores, index=df.index, name="Behavioral_Score_Final")


__all__ = ["compute_behavioral_score"]
