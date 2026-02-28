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
from services.explanation_generator import generate_fraud_summary


st.set_page_config(
    page_title="Review Fraud Detection Dashboard",
    layout="wide"
)


def main():

    # ---------------------------
    # UI SECTIONS
    # ---------------------------
    load_styles()
    render_header()
    render_intro_sections()

    # ---------------------------
    # FILE UPLOAD
    # ---------------------------
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

    # ---------------------------
    # SESSION STATE
    # ---------------------------
    if "processed" not in st.session_state:
        st.session_state.processed = False

    if st.button("Preprocess Data"):
        st.session_state.processed = True

    # ======================================================
    # PREPROCESS + TRAIN (RUNS ONLY ONCE)
    # ======================================================
    if st.session_state.processed and "training_df" not in st.session_state:

        textual_df, vectorizer = textual_training_dataset(reviews_df, products_df)
        product_temporal_df, customer_temporal_df = temporal_features(reviews_df)
        product_behavior_df = product_behavioral_features(reviews_df)
        customer_behavior_df = customer_behavioral_features(users_df, reviews_df)

        # DISPLAY PREPROCESSED DATA
        st.markdown("<div class='dataset-heading'>Textual Features</div>", unsafe_allow_html=True)
        st.dataframe(textual_df.head(2), use_container_width=True)
        st.divider()

        st.markdown("<div class='dataset-heading'>Product Behavioral Features</div>", unsafe_allow_html=True)
        st.dataframe(product_behavior_df.head(2), use_container_width=True)
        st.divider()

        st.markdown("<div class='dataset-heading'>Customer Behavioral Features</div>", unsafe_allow_html=True)
        st.dataframe(customer_behavior_df.head(2), use_container_width=True)
        st.divider()

        st.markdown("<div class='dataset-heading'>Product Temporal Features</div>", unsafe_allow_html=True)
        st.dataframe(product_temporal_df.head(2), use_container_width=True)
        st.divider()

        st.markdown("<div class='dataset-heading'>Customer Temporal Features</div>", unsafe_allow_html=True)
        st.dataframe(customer_temporal_df.head(2), use_container_width=True)
        st.divider()

        # BUILD TRAINING DATASET
        training_df = build_training_dataset(
            textual_df,
            reviews_df,
            customer_behavior_df,
            product_behavior_df,
            product_temporal_df,
            customer_temporal_df
        )

        # TRAIN MODEL
        model, accuracy, roc_auc = train_xgboost_model(training_df)

        # SAVE TO SESSION
        st.session_state.training_df = training_df
        st.session_state.model_saved = load_model()
        st.session_state.reviews_df = reviews_df
        st.session_state.users_df = users_df
        st.session_state.products_df = products_df
        st.session_state.orders_df = orders_df
        st.session_state.platforms_df = platforms_df
        st.session_state.accuracy = accuracy
        st.session_state.roc_auc = roc_auc

        # MODEL DETAILS
        st.markdown("""
        <div class="section-heading">Model Details</div>
        <div class="section-underline"></div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <ul class="section-text model-list">
            <li><b>Model Trained :</b> XGBoost Classifier</li>
            <li><b>Accuracy :</b> {accuracy * 100:.2f}%</li>
            <li><b>ROC-AUC Score :</b> {roc_auc:.4f}</li>
            <li><b>Number of Features Used :</b> {training_df.shape[1]}</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

    # ======================================================
    # FRAUD RISK INSPECTION (NO RETRAINING)
    # ======================================================
    if "training_df" in st.session_state:

        training_df = st.session_state.training_df
        saved = st.session_state.model_saved
        reviews_df = st.session_state.reviews_df
        users_df = st.session_state.users_df
        products_df = st.session_state.products_df
        orders_df = st.session_state.orders_df
        platforms_df = st.session_state.platforms_df

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

            st.markdown(
                "<div class='dataset-heading'>Customer Details</div>",
                unsafe_allow_html=True
            )
            st.text_input("Account Created", value=_val(user_row, "account_created", "Account_Created"), disabled=True, key="ui_account_created")

            st.divider()

            # ======================================================
            # DYNAMIC SUMMARY (generated from feature values & importance)
            # ======================================================
            st.markdown("""
            <div class="section-heading">Summary</div>
            <div class="section-underline"></div>
            """, unsafe_allow_html=True)

            feature_importance = model.feature_importances_
            summary_text = generate_fraud_summary(
                selected_row,
                feature_list,
                feature_importance,
                risk_score,
                prediction,
            )
            st.markdown(summary_text)


if __name__ == "__main__":
    main()