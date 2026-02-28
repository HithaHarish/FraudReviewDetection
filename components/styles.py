import streamlit as st

def load_styles():
    st.markdown("""
        <style>

        /* Remove ALL top padding inside sidebar */
        section[data-testid="stSidebar"] > div {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }

        section[data-testid="stSidebar"] div:first-child {
            margin-top: 0rem !important;
            padding-top: 0rem !important;
        }

        .block-container {
            padding-top: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        .main {
            background-color: #F8F5EE;
        }

        .header-title {
            font-size: 30px;
            font-weight: 500;
            color: black;
            margin-top: 0px;
            margin-bottom: 14px;
        }

        .section-heading {
            font-size: 20px;
            font-weight: 400;
            color: black;
            margin-bottom: 6px;
            margin-top: 18px;
        }

        .section-underline {
            width: 80px;
            height: 3px;
            background-color: #A5D6A7;
            margin-bottom: 18px;
        }

        .section-text {
            font-size: 16px;
            font-weight: 300;
            line-height: 1.85;
            color: black;
        }

        /* NEW: Model Body Style */
        .model-body {
            font-size: 17px;
            font-weight: 300;
            line-height: 1.9;
            color: black;
        }

        /* NEW: Dataset Heading */
        .dataset-heading {
            font-size: 20px;
            font-weight: 400;
            color: black;
            margin-top: 18px;
            margin-bottom: 8px;
        }

        ul.custom-list {
            padding-left: 15px;
            margin-top: 6px;
        }

        ul.model-list {
            padding-left: 22px;
            margin-top: 6px;
        }

        ul.custom-list li {
            margin-bottom: 26px;
            line-height: 1.9;
        }

        ul.model-list li {
            margin-bottom: 8px;
            line-height: 1.7;
        }

        section[data-testid="stSidebar"] {
            width: 400px !important;
            min-width: 400px !important;
            max-width: 420px !important;
            padding-top: 0.5rem !important;
        }

        </style>
        """, unsafe_allow_html=True)