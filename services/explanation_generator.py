"""
Dynamic explanation generator for fraud risk scores.
Generates natural language summaries based on actual feature values,
feature importance, and prediction outcome — no hardcoded explanations.
"""

import numpy as np
import pandas as pd

# Feature metadata: (human_name, fraud_direction, value_interpretation)
# fraud_direction: "high" = higher value increases fraud risk, "low" = lower value increases fraud risk
# value_interpretation: callable(value) -> str or None for default
FEATURE_METADATA = {
    # Textual
    "structural_score": ("structural irregularity", "high", lambda v: "excessive capitals/punctuation" if v > 0.5 else "unusual formatting"),
    "repetition_score": ("lexical repetition", "high", lambda v: "repeated words" if v > 0.3 else "low word diversity"),
    "sentiment_score": ("sentiment extremity", "high", lambda v: "polarized sentiment" if abs(v) > 0.5 else "moderate sentiment"),
    "review_length": ("review length", "low", lambda v: "very short" if v < 10 else "unusually brief"),
    "capital_ratio": ("capital usage", "high", lambda v: "excessive capitals" if v > 0.3 else "normal capitalization"),
    "punctuation_density": ("punctuation density", "high", lambda v: "excessive punctuation" if v > 0.05 else "normal punctuation"),
    "promotional_intensity": ("promotional density", "high", lambda v: "promotional keywords" if v > 0.3 else "hype-like phrasing"),
    # Customer behavioral
    "verified_review_ratio": ("verified purchase rate", "low", lambda v: "low verification" if v < 0.5 else "high verification"),
    "refund_ratio": ("refund rate", "high", lambda v: "high refund rate" if v > 0.2 else "refund history"),
    "account_age_days": ("account age", "low", lambda v: "new account" if v < 90 else "established account"),
    "review_frequency_per_week": ("review frequency", "high", lambda v: "unusually frequent" if v > 5 else "high activity"),
    "cross_platform_same_product_count": ("cross-platform review count", "high", lambda v: "same product on multiple platforms" if v > 0 else None),
    "same_platform_repeat_count": ("repeat review count", "high", lambda v: "duplicate reviews" if v > 0 else None),
    "positive_review_ratio": ("positive review ratio", "high", lambda v: "exclusively positive" if v > 0.9 else None),
    "avg_rating": ("average rating", "high", lambda v: "consistently high ratings" if v >= 4.5 else None),
    # Product behavioral
    "refund_rating_mismatch_ratio": ("refund-rating mismatch", "high", lambda v: "high refunds despite high ratings" if v > 0.2 else None),
    "cross_platform_customer_ratio": ("cross-platform customer ratio", "high", lambda v: "same customers across platforms" if v > 0.3 else None),
    "same_platform_repeat_ratio": ("repeat review ratio", "high", lambda v: "duplicate reviews" if v > 0.1 else None),
    "review_spike_count": ("review spike count", "high", lambda v: "burst of reviews" if v > 0 else None),
    "unique_review_ratio": ("unique review ratio", "low", lambda v: "low uniqueness" if v < 0.5 else None),
    # Temporal
    "review_burst_count": ("review burst activity", "high", lambda v: "burst posting" if v > 0 else None),
    "coordinated_multi_user_events": ("coordinated activity", "high", lambda v: "coordinated multi-user" if v > 0 else None),
    "night_posting_ratio": ("night posting ratio", "high", lambda v: "unusual posting hours" if v > 0.3 else None),
    "sudden_surge_ratio": ("sudden surge", "high", lambda v: "sudden review surge" if v > 2 else None),
    "customer_burst_count": ("customer burst", "high", lambda v: "burst activity" if v > 0 else None),
    "customer_night_ratio": ("customer night ratio", "high", lambda v: "late-night posting" if v > 0.3 else None),
    "avg_time_gap_seconds": ("avg time gap", "low", lambda v: "suspiciously regular gaps" if 0 < v < 60 else None),
    "gap_variance": ("gap variance", "low", lambda v: "low variance" if v < 100 and v > 0 else None),
}


def _get_feature_info(feature_name: str) -> tuple:
    """Get (human_name, fraud_direction, interp_fn) for a feature."""
    for key, val in FEATURE_METADATA.items():
        if key in feature_name or feature_name in key:
            return val
    # Fallback: derive from name
    human = feature_name.replace("_", " ").title()
    return (human, "high", lambda v: None)


def _contributes_to_fraud(feature_name: str, value: float, direction: str) -> bool:
    """Determine if this feature value contributes to fraud risk."""
    v = abs(value) if "sentiment" in feature_name else value
    # Features with large scale (days, counts) - use raw thresholds
    if "account_age" in feature_name:
        return v < 90  # new account
    if "total_reviews" in feature_name or "burst" in feature_name or "spike" in feature_name:
        return v > 5 if direction == "high" else v < 2
    if direction == "high":
        return v > 0.3
    if direction == "low":
        return v < 0.5
    return False


def _contributes_to_safety(feature_name: str, value: float, direction: str) -> bool:
    """Determine if this feature value contributes to safety."""
    v = abs(value) if "sentiment" in feature_name else value
    if "account_age" in feature_name:
        return v >= 180  # established account
    if direction == "high":
        return v <= 0.2
    if direction == "low":
        return v >= 0.7
    return False


def generate_fraud_summary(
    selected_row: pd.DataFrame,
    feature_names: list,
    feature_importance: np.ndarray,
    risk_score: float,
    prediction: int,
) -> str:
    """
    Generate a dynamic natural language summary explaining why the review
    was scored as fraud or safe, based on actual feature values and importance.
    """
    row = selected_row.iloc[0]
    paragraphs = []

    # Build importance-weighted feature list
    imp_list = list(zip(feature_names, feature_importance))
    imp_list.sort(key=lambda x: x[1], reverse=True)
    top_n = imp_list[:12]

    fraud_reasons = []
    safety_reasons = []

    for feat_name, imp in top_n:
        if feat_name not in row.index:
            continue
        try:
            val = float(row[feat_name])
        except (TypeError, ValueError):
            continue

        human_name, direction, interp_fn = _get_feature_info(feat_name)
        interp = interp_fn(val) if interp_fn else None
        desc = interp or human_name

        if imp < 0.01:
            continue

        if _contributes_to_fraud(feat_name, val, direction):
            fraud_reasons.append((human_name, desc, val, imp))
        elif _contributes_to_safety(feat_name, val, direction):
            safety_reasons.append((human_name, desc, val, imp))

    # Also consider raw values for borderline cases
    for feat_name, imp in top_n:
        if feat_name not in row.index:
            continue
        try:
            val = float(row[feat_name])
        except (TypeError, ValueError):
            continue

        human_name, direction, interp_fn = _get_feature_info(feat_name)
        if (human_name, None, val, imp) in [(r[0], r[1], r[2], r[3]) for r in fraud_reasons + safety_reasons]:
            continue

        if direction == "high" and val > 0.5 and imp > 0.02:
            fraud_reasons.append((human_name, interp_fn(val) if interp_fn else human_name, val, imp))
        elif direction == "low" and val < 0.3 and imp > 0.02:
            fraud_reasons.append((human_name, interp_fn(val) if interp_fn else human_name, val, imp))
        elif direction == "high" and val < 0.2 and imp > 0.02:
            safety_reasons.append((human_name, "normal range", val, imp))
        elif direction == "low" and val > 0.7 and imp > 0.02:
            safety_reasons.append((human_name, "normal range", val, imp))

    # Deduplicate by human_name
    seen_f = set()
    seen_s = set()
    fraud_reasons = [r for r in fraud_reasons if r[0] not in seen_f and not seen_f.add(r[0])]
    safety_reasons = [r for r in safety_reasons if r[0] not in seen_s and not seen_s.add(r[0])]

    risk_pct = risk_score * 100
    is_flagged = prediction == 1 or risk_pct >= 50

    if is_flagged:
        paragraphs.append(
            f"This review received a fraud risk score of {risk_pct:.1f}% and was flagged as suspicious."
        )
        if fraud_reasons:
            paragraphs.append(
                "**The following factors contributed to the elevated risk:**"
            )
            for i, (name, desc, val, _) in enumerate(fraud_reasons[:5], 1):
                paragraphs.append(f"- **{name}**: {desc} (value: {val:.3f})")
        else:
            paragraphs.append(
                "The model flagged this review based on combined signals across textual, behavioral, and temporal features."
            )
        if safety_reasons:
            paragraphs.append(
                "**Some mitigating factors were also observed:**"
            )
            for i, (name, desc, val, _) in enumerate(safety_reasons[:3], 1):
                paragraphs.append(f"- **{name}**: {desc} (value: {val:.3f})")
    else:
        paragraphs.append(
            f"This review received a fraud risk score of {risk_pct:.1f}% and was considered safe."
        )
        if safety_reasons:
            paragraphs.append(
                "**The following factors supported a safe classification:**"
            )
            for i, (name, desc, val, _) in enumerate(safety_reasons[:5], 1):
                paragraphs.append(f"- **{name}**: {desc} (value: {val:.3f})")
        if fraud_reasons:
            paragraphs.append(
                "**Some minor risk indicators were present but did not outweigh overall safety:**"
            )
            for i, (name, desc, val, _) in enumerate(fraud_reasons[:3], 1):
                paragraphs.append(f"- **{name}**: {desc} (value: {val:.3f})")
        if not safety_reasons and not fraud_reasons:
            paragraphs.append(
                "Feature values across textual, behavioral, and temporal layers fell within normal ranges."
            )

    return "\n\n".join(paragraphs)
