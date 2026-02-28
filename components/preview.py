import streamlit as st

def show_dataset_preview(
    users_df,
    products_df,
    orders_df,
    reviews_df,
    platforms_df
):
    """
    Displays first 2 rows of each uploaded dataset.
    Pure UI function — no preprocessing logic.
    """

    st.markdown(
        "<div class='dataset-heading'>Uploaded Dataset Preview</div>",
        unsafe_allow_html=True
    )

    datasets = {
        "Users Dataset": users_df,
        "Products Dataset": products_df,
        "Orders Dataset": orders_df,
        "Reviews Dataset": reviews_df,
        "Platforms Dataset": platforms_df,
    }

    for name, df in datasets.items():

        st.markdown(
            f"<div class='dataset-heading'>{name}</div>",
            unsafe_allow_html=True
        )

        if df is not None and not df.empty:
            st.dataframe(
                df.head(2),
                use_container_width=True
            )
        else:
            st.warning(f"{name} is empty or not loaded properly.")

        st.divider()