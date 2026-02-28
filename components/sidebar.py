import streamlit as st
def render_sidebar():
    st.sidebar.markdown("### Upload CSV Files")

    users_file = st.sidebar.file_uploader("Users Dataset (CSV)", type=["csv"])
    products_file = st.sidebar.file_uploader("Products Dataset (CSV)", type=["csv"])
    orders_file = st.sidebar.file_uploader("Orders Dataset (CSV)", type=["csv"])
    reviews_file = st.sidebar.file_uploader("Reviews Dataset (CSV)", type=["csv"])
    platforms_file = st.sidebar.file_uploader("Platforms Dataset (CSV)", type=["csv"])

    return users_file, products_file, orders_file, reviews_file, platforms_file