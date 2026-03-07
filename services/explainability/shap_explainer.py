import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None


def create_explainer(model):
    """
    Create and return a SHAP TreeExplainer for the given model.
    Returns None if SHAP is not available or the model is unsupported.
    """
    if shap is None or model is None:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return None


def _extract_shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """
    Return a 1D SHAP vector for a single-row input X.
    Handles SHAP's different return shapes for tree models.
    """
    shap_values = explainer.shap_values(X)

    # Some explainers return list-of-arrays (per class)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]  # positive class for binary

    values = np.asarray(shap_values)
    if values.ndim == 2:
        values = values[0]
    return values


def get_local_explanation(model, selected_row: pd.DataFrame, feature_list, risk_score=None):
    """
    Local explanation for one selected review using SHAP TreeExplainer.

    Keeps only features with shap_value > 0 (contributing toward fraud),
    sorts by |shap_value| descending, returns top 5.
    """
    explainer = create_explainer(model)
    if explainer is None:
        return None

    if selected_row is None or selected_row.empty:
        return None

    # Ensure we work with a single row aligned to feature_list
    try:
        row_features = selected_row[feature_list].iloc[0]
    except Exception:
        return None

    X = np.asarray(row_features.values, dtype=float).reshape(1, -1)

    try:
        shap_vec = _extract_shap_values(explainer, X)
    except Exception:
        return None

    top = []
    for fname, fval, sval in zip(feature_list, row_features.values, shap_vec):
        if float(sval) > 0:
            top.append(
                {
                    "feature": str(fname),
                    "feature_value": fval,
                    "shap_impact": float(sval),
                }
            )

    top.sort(key=lambda d: abs(d["shap_impact"]), reverse=True)
    top = top[:3]

    return {
        "risk_score": float(risk_score) if risk_score is not None else None,
        "top_features": top,
    }


def build_reasoning_summary(explanation_data: dict) -> str:
    """
    Build a human-readable summary strictly from SHAP contributions.
    No hardcoded feature names; works for arbitrary features.
    """
    if not explanation_data:
        return "No SHAP explanation available for this review."

    top_features = explanation_data.get("top_features") or []
    risk_score = explanation_data.get("risk_score")

    header = []
    if risk_score is not None:
        header.append(f"Fraud Risk Score: {risk_score * 100:.2f}%")
        header.append("")

    if not top_features:
        return "\n".join(
            header
            + [
                "Top Contributing Parameters: (none with positive SHAP contribution)",
                "",
                "Summary:",
                "This review was scored based on combined signals, but no single feature had a positive SHAP contribution strong enough to list.",
            ]
        )

    max_imp = max(abs(f["shap_impact"]) for f in top_features) or 1.0

    bullets = []
    for f in top_features:
        name = str(f["feature"]).replace("_", " ").strip()
        value = f["feature_value"]
        impact = float(f["shap_impact"])

        rel = abs(impact) / max_imp
        if rel >= 0.66:
            strength = "Strong indicator"
        elif rel >= 0.33:
            strength = "Moderate indicator"
        else:
            strength = "Low indicator"

        bullets.append(
            f"- {strength}: {name} ({value}) increased fraud likelihood (SHAP +{impact:.4f})."
        )

    return "\n".join(
        header
        + [
            "Summary:",
            "This review was flagged as suspicious due to multiple SHAP-identified contributors:",
            *bullets,
        ]
    )

