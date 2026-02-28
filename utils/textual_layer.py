from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


RANDOM_STATE = 42


def _scale_existing_probability(series: pd.Series) -> pd.Series:
    """
    Scale an existing probability-like series to 0–100.
    Handles:
    - Values already in [0,1]
    - Values in arbitrary ranges via MinMax scaling
    """
    s = series.astype(float)
    if s.max() <= 1.0 and s.min() >= 0.0:
        return s * 100.0
    scaler = MinMaxScaler(feature_range=(0, 100))
    return pd.Series(
        scaler.fit_transform(s.to_numpy().reshape(-1, 1)).reshape(-1),
        index=series.index,
    )


def _infer_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to infer a binary label column for supervised training.
    Common possibilities: 'Fraud_Label', 'Label', 'Is_Fraud', 'is_fraud'.
    """
    candidates = ["Fraud_Label", "Label", "Is_Fraud", "is_fraud"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _heuristic_text_risk(review_texts: pd.Series) -> pd.Series:
    """
    Fallback heuristic textual risk when no labels are available.
    Combines:
    - Review length
    - Punctuation density (e.g. '!!!')
    - Uppercase ratio
    """
    texts = review_texts.fillna("").astype(str)
    lengths = texts.str.len().replace(0, 1)
    exclam = texts.str.count("!")
    upper_ratio = texts.apply(
        lambda s: sum(1 for ch in s if ch.isupper()) / len(s) if len(s) > 0 else 0.0
    )
    raw = 0.5 * (lengths / lengths.max()) + 0.3 * (exclam / (exclam.max() or 1)) + 0.2 * upper_ratio
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled = scaler.fit_transform(raw.to_numpy().reshape(-1, 1)).reshape(-1)
    return pd.Series(scaled, index=review_texts.index)


def compute_text_score(df: pd.DataFrame, text_column: str = "Review_Text") -> Tuple[pd.DataFrame, Optional[LogisticRegression]]:
    """
    Compute textual fraud score (0–100) for each review.

    Strategy:
    1. If 'Text_Fraud_Probability' exists and is non-null:
       - Scale to 0–100 and use as Text_Score.
    2. Else:
       - If a label column is available, train TF-IDF + LogisticRegression
         to predict fraud probability.
       - If no label column exists, fall back to a heuristic textual risk.

    Returns:
        (updated_df, trained_model_or_None)
    """
    df = df.copy()

    # Case 1: existing probability column
    if "Text_Fraud_Probability" in df.columns and df["Text_Fraud_Probability"].notna().any():
        df["Text_Score"] = _scale_existing_probability(df["Text_Fraud_Probability"].fillna(0.0))
        return df, None

    if text_column not in df.columns:
        warnings.warn(f"Text column '{text_column}' not found. Filling Text_Score with zeros.", RuntimeWarning)
        df["Text_Score"] = 0.0
        return df, None

    texts = df[text_column].fillna("").astype(str)

    label_col = _infer_label_column(df)
    if label_col is None:
        # Heuristic fallback; no labels for supervised learning
        warnings.warn(
            "No label column found for textual model; using heuristic textual risk instead.",
            RuntimeWarning,
        )
        df["Text_Score"] = _heuristic_text_risk(texts)
        return df, None

    # Supervised TF-IDF + LogisticRegression
    y = df[label_col].astype(int)

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Compute probabilities for all reviews
    proba = model.predict_proba(X)[:, 1]  # probability of fraud class
    df["Text_Score"] = proba * 100.0

    # Persist vectorizer on the model for potential reuse
    model._vectorizer = vectorizer  # type: ignore[attr-defined]

    return df, model


def compute_textual_score(
    df: pd.DataFrame,
    text_column: str = "Review_Text",
    random_state: int | None = None,
) -> pd.Series:
    """
    Backwards-compatible wrapper that matches the expected API in fraud_model:
    returns just the Text_Score series (0–100) for each row.
    """
    # random_state is accepted for compatibility but currently unused,
    # since the heuristic/text model already uses an internal fixed seed.
    _ = random_state
    df_with_scores, _ = compute_text_score(df, text_column=text_column)
    if "Text_Score" not in df_with_scores.columns:
        return pd.Series(0.0, index=df.index, name="Text_Score")
    return df_with_scores["Text_Score"]


__all__ = ["compute_text_score", "compute_textual_score"]

