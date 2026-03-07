import os

# Load .env so OPENAI_API_KEY is available (env and .env override Streamlit secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import pandas as pd

from components.styles import load_styles
from components.header import render_header
from components.sections import render_intro_sections
from components.sidebar import render_sidebar
from components.preview import show_dataset_preview

from services.preprocessing.textual import textual_training_dataset
from services.preprocessing.temporal import temporal_features
from services.preprocessing.behavioral_product import product_behavioral_features
from services.preprocessing.behavioral_user import customer_behavioral_features
from services.models.xgboost_model import train_xgboost_model
from services.build_training_dataset import build_training_dataset
from services.models.predict import load_model, predict_review
from services.explainability.shap_explainer import (
    get_local_explanation,
    build_reasoning_summary,
)
from services.explainability.genai_reasoning import generate_llm_explanation
from services.formatting.format import format_llm_output

st.set_page_config(
page_title="Review Fraud Detection Dashboard",
layout="wide"
)

def main():
    load_styles()
    render_header()
    render_intro_sections()

    users_file, products_file, orders_file, reviews_file, platforms_file = render_sidebar()

    if not all([users_file, products_file, orders_file, reviews_file, platforms_file]):
        st.info("Upload all 5 datasets to continue.")
        return

    users_df = pd.read_csv(users_file)
    products_df = pd.read_csv(products_file)
    orders_df = pd.read_csv(orders_file)
    reviews_df = pd.read_csv(reviews_file)
    platforms_df = pd.read_csv(platforms_file)

    show_dataset_preview(users_df, products_df, orders_df, reviews_df, platforms_df)

    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "preprocessed_samples" not in st.session_state:
        st.session_state.preprocessed_samples = None
    if "model_saved" not in st.session_state:
        st.session_state.model_saved = False
    if "training_df" not in st.session_state:
        st.session_state.training_df = None
    if "accuracy" not in st.session_state:
        st.session_state.accuracy = 0.0
    if "roc_auc" not in st.session_state:
        st.session_state.roc_auc = 0.0

    if st.button("Preprocess Data"):
        st.session_state.processed = True

    if st.session_state.processed and st.session_state.training_df is None:

        textual_df, vectorizer = textual_training_dataset(reviews_df, products_df)
        product_temporal_df, customer_temporal_df = temporal_features(reviews_df)
        product_behavior_df = product_behavioral_features(reviews_df)
        customer_behavior_df = customer_behavioral_features(users_df, reviews_df)

        st.session_state.preprocessed_samples = {
            "Textual Features": textual_df.head(2),
            "Product Behavioral Features": product_behavior_df.head(2),
            "Customer Behavioral Features": customer_behavior_df.head(2),
            "Product Temporal Features": product_temporal_df.head(2),
            "Customer Temporal Features": customer_temporal_df.head(2),
        }

        training_df = build_training_dataset(
            textual_df,
            reviews_df,
            customer_behavior_df,
            product_behavior_df,
            product_temporal_df,
            customer_temporal_df
        )

        model, accuracy, roc_auc = train_xgboost_model(training_df)

        st.session_state.training_df = training_df
        st.session_state.model_saved = load_model()
        st.session_state.reviews_df = reviews_df
        st.session_state.users_df = users_df
        st.session_state.products_df = products_df
        st.session_state.orders_df = orders_df
        st.session_state.platforms_df = platforms_df
        st.session_state.accuracy = accuracy
        st.session_state.roc_auc = roc_auc

        st.session_state.processed = True

    if st.session_state.preprocessed_samples:
        with st.expander("Preprocessed Feature Samples", expanded=False):
            for title, df in st.session_state.preprocessed_samples.items():
                st.markdown(f"<div class='dataset-heading'>{title}</div>", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True)
                st.divider()

    if st.session_state.training_df is not None:
        st.markdown("""
        <div class="section-heading">Model Details</div>
        <div class="section-underline"></div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <ul class="section-text model-list">
            <li><b>Model Trained :</b> XGBoost Classifier</li>
            <li><b>Accuracy :</b> {st.session_state.accuracy * 100:.2f}%</li>
            <li><b>ROC-AUC Score :</b> {st.session_state.roc_auc:.4f}</li>
            <li><b>Number of Features Used :</b> {st.session_state.training_df.shape[1]}</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    if st.session_state.training_df is not None and st.session_state.model_saved:

        training_df = st.session_state.training_df
        saved = st.session_state.model_saved
        reviews_df = st.session_state.get("reviews_df")
        users_df = st.session_state.get("users_df")
        products_df = st.session_state.get("products_df")
        orders_df = st.session_state.get("orders_df")
        platforms_df = st.session_state.get("platforms_df")

        model = saved["model"]
        feature_list = saved["features"]

        st.markdown("""
        <div class="section-heading">Fraud Risk Inspection</div>
        <div class="section-underline"></div>
        """, unsafe_allow_html=True)

        review_ids = training_df["review_id"].unique()

        selected_review_id = st.selectbox(
            "Select Review",
            review_ids,
            key="review_selector"
        )

        selected_row = training_df[
            training_df["review_id"] == selected_review_id
        ]

        if not selected_row.empty:

            selected_row = selected_row.copy()

            customer_id = selected_row["customer_id"].values[0]
            product_id = selected_row["product_id"].values[0]

            def _col(df, *candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            def _val(row, *candidates, default="-"):
                if row is None:
                    return default
                for c in candidates:
                    if c in row.index:
                        v = row[c]
                        return "-" if pd.isna(v) else str(v)
                return default

            rid_col = _col(reviews_df, "review_id", "Review_ID") or "review_id"
            product_col = _col(products_df, "product_id", "Product_ID") or "product_id"
            customer_col = _col(users_df, "customer_id", "Customer_ID") or "customer_id"

            review_row = reviews_df[reviews_df[rid_col] == selected_review_id]
            review_row = review_row.iloc[0] if not review_row.empty else None

            product_row = products_df[products_df[product_col] == product_id]
            product_row = product_row.iloc[0] if not product_row.empty else None

            user_row = users_df[users_df[customer_col] == customer_id]
            user_row = user_row.iloc[0] if not user_row.empty else None

            platform_name = ""
            if review_row is not None:
                pid_col = _col(reviews_df, "platform_id", "Platform_ID")
                if pid_col:
                    pid = review_row[pid_col]
                    plat_id_col = _col(platforms_df, "platform_id", "Platform_ID")
                    plat_name_col = _col(platforms_df, "platform_name", "Platform_Name")
                    if plat_id_col and plat_name_col:
                        plat = platforms_df[platforms_df[plat_id_col].astype(str) == str(pid)]
                        if not plat.empty:
                            platform_name = plat.iloc[0][plat_name_col]

            col3, col4 = st.columns([3, 1])

            with col3:
                original_review = _val(review_row, "review_text", "Review_Text", default="")
                st.text_area("Review Text", original_review, height=150)

            prediction, risk_score = predict_review(
                model,
                feature_list,
                selected_row
            )

            with col4:
                st.metric("Fraud Risk Score", f"{risk_score * 100:.2f}%")

                # --------------------------------------------------
                # FRAUD INTERPRETATION (simple text format)
                # --------------------------------------------------
                if risk_score < 0.30:
                    label = "Genuine Review"
                    reason = "0–30%: Genuine range"
                elif risk_score < 0.60:
                    label = "Suspicious Review"
                    reason = "30–60%: Suspicious range"
                else:
                    label = "Fraudulent Review"
                    reason = "60–100%: Fraud range"

                st.markdown(f"**Prediction:** {label}")
                st.markdown(f"({reason})")

            st.divider()

            # ---------------------------
            # REVIEW DETAILS
            # ---------------------------
            st.markdown(
                "<div class='dataset-heading'>Review Details</div>",
                unsafe_allow_html=True
            )
            r1, r2 = st.columns(2)
            with r1:
                st.text_input("Rating", value=_val(review_row, "rating", "Rating"), disabled=True, key="ri_rating")
                st.text_input("Verified Purchase", value=_val(review_row, "verified_purchase", "Verified_Purchase"), disabled=True, key="ri_verified")
            with r2:
                st.text_input("Platform", value=platform_name or _val(review_row, "platform_id", "Platform_ID"), disabled=True, key="ri_platform")
                st.text_input("Refunded", value=_val(review_row, "refunded_product", "Refunded_Product"), disabled=True, key="ri_refunded")
            if "rating_sentiment_mismatch" in selected_row.columns:
                try:
                    mm = int(selected_row["rating_sentiment_mismatch"].values[0])
                    st.text_input("Rating–Sentiment Mismatch", value="Yes" if mm == 1 else "No", disabled=True, key="ri_rating_mismatch")
                except Exception:
                    pass

            # ---------------------------
            # PRODUCT DETAILS
            # ---------------------------
            st.markdown(
                "<div class='dataset-heading'>Product Details</div>",
                unsafe_allow_html=True
            )
            p1, p2 = st.columns(2)
            with p1:
                st.text_input("Name", value=_val(product_row, "name", "Name", "product_name"), disabled=True, key="pi_name")
                st.text_input("Brand", value=_val(product_row, "brand", "Brand"), disabled=True, key="pi_brand")
            with p2:
                st.text_input("Category", value=_val(product_row, "category", "Category"), disabled=True, key="pi_category")

            # ---------------------------
            # CUSTOMER DETAILS
            # ---------------------------
            st.markdown(
                "<div class='dataset-heading'>Customer Details</div>",
                unsafe_allow_html=True
            )
            st.text_input("Account Created", value=_val(user_row, "account_created", "Account_Created"), disabled=True, key="ui_account_created")

            st.divider()

            # ======================================================
            # TOP CONTRIBUTING PARAMETERS + SHAP (summary) — after details
            # ======================================================

            if risk_score > 0.0:

                explanation_data = get_local_explanation(
                    model=model,
                    selected_row=selected_row,
                    feature_list=feature_list,
                    risk_score=risk_score,
                )

                if explanation_data is not None and explanation_data.get("top_features"):

                    st.markdown(
                        "<div class='dataset-heading'>Top Contributing Parameters</div>",
                        unsafe_allow_html=True,
                    )

                    top_df = pd.DataFrame(
                        [
                            {
                                "Feature": f["feature"].replace("_", " ").title(),
                                "Value": f["feature_value"],
                                "Impact": f["shap_impact"],
                            }
                            for f in explanation_data["top_features"]
                        ]
                    )

                    st.dataframe(top_df, use_container_width=True)

                    st.markdown(
                        "<div class='dataset-heading'>LLM-based Explanation</div>",
                        unsafe_allow_html=True,
                    )

                    evidence = {
                        "rating": _val(review_row, "rating", "Rating"),
                        "verified_purchase": _val(review_row, "verified_purchase", "Verified_Purchase"),
                        "refunded_product": _val(review_row, "refunded_product", "Refunded_Product"),
                        "sentiment_score": float(selected_row.get("sentiment_score", 0)),
                        "sentiment_intensity": float(selected_row.get("sentiment_intensity", 0)),
                        "product_relevance": float(selected_row.get("product_relevance_score", 0)),
                        "rating_sentiment_mismatch": int(selected_row.get("rating_sentiment_mismatch", 0)),
                        "punctuation_density": float(selected_row.get("punctuation_density", 0)),
                        "review_length": int(selected_row.get("review_length", 0))
                    }
                    
                    # Prefer env / .env over Streamlit secrets (so export or .env is used)
                    summary = generate_llm_explanation(
                        review_text=original_review,
                        risk_score=risk_score,
                        top_features=explanation_data["top_features"],
                        prediction_label=label,
                        evidence=evidence
                    )
                    if summary:
                        formatted_summary = format_llm_output(summary, original_review)

                        st.markdown(formatted_summary)

                    else:
                        st.markdown(build_reasoning_summary(explanation_data))

if __name__ == "__main__":
    main()
