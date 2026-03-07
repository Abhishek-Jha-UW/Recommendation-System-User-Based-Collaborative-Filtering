import streamlit as st
import pandas as pd
from io import StringIO
from model import (
    compute_user_similarity,
    recommend_user_based,
    compute_item_similarity,
    recommend_item_based,
    run_market_basket,
    recommend_market_basket
)

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Recommendation System",
    layout="wide"
)

st.title("Recommendation System")
st.markdown(
    """
This application supports three recommendation methods:
- User-Based Collaborative Filtering  
- Item-Based Collaborative Filtering  
- Market Basket Analysis  

You may upload your own dataset or use the provided sample dataset.
"""
)

# ---------------------------------------------------------
# Sample Dataset
# ---------------------------------------------------------
sample_data = pd.DataFrame({
    "user_id": ["U1", "U1", "U2", "U2", "U3", "U3"],
    "item_id": ["A", "B", "A", "C", "B", "D"],
    "rating": [5, 3, 4, 2, 5, 4]
})

# Template CSV for download
template_csv = """user_id,item_id,rating
U1,A,5
U1,B,3
U2,A,4
U2,C,2
U3,B,5
U3,D,4
"""

st.download_button(
    label="Download Template CSV",
    data=template_csv,
    file_name="recommendation_template.csv",
    mime="text/csv"
)

# ---------------------------------------------------------
# Dataset Upload Section
# ---------------------------------------------------------
st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file with columns: user_id, item_id, rating",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully.")
else:
    st.info("Using sample dataset.")
    df = sample_data.copy()

st.write("Preview of dataset:")
st.dataframe(df.head())

# ---------------------------------------------------------
# Method Selection
# ---------------------------------------------------------
st.subheader("Select Recommendation Method")

method = st.selectbox(
    "Choose a method:",
    ["User-Based Collaborative Filtering", "Item-Based Collaborative Filtering", "Market Basket Analysis"]
)

# ---------------------------------------------------------
# USER-BASED CF
# ---------------------------------------------------------
if method == "User-Based Collaborative Filtering":
    st.subheader("User-Based Collaborative Filtering")

    try:
        user_item_matrix, similarity_df = compute_user_similarity(df)

        user_list = list(user_item_matrix.index)
        selected_user = st.selectbox("Select User:", user_list)

        if st.button("Generate Recommendations"):
            recs = recommend_user_based(
                user_id=selected_user,
                user_item_matrix=user_item_matrix,
                similarity_df=similarity_df,
                k=5
            )

            if recs:
                st.write("Top Recommendations:")
                st.table(pd.DataFrame({"Recommended Items": recs}))
            else:
                st.info("No recommendations available for this user.")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------------
# ITEM-BASED CF
# ---------------------------------------------------------
elif method == "Item-Based Collaborative Filtering":
    st.subheader("Item-Based Collaborative Filtering")

    try:
        user_item_matrix, item_similarity_df = compute_item_similarity(df)

        user_list = list(user_item_matrix.index)
        selected_user = st.selectbox("Select User:", user_list)

        if st.button("Generate Recommendations"):
            recs = recommend_item_based(
                user_id=selected_user,
                user_item_matrix=user_item_matrix,
                item_similarity_df=item_similarity_df,
                k=5
            )

            if recs:
                st.write("Top Recommendations:")
                st.table(pd.DataFrame({"Recommended Items": recs}))
            else:
                st.info("No recommendations available for this user.")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------------
# MARKET BASKET ANALYSIS
# ---------------------------------------------------------
elif method == "Market Basket Analysis":
    st.subheader("Market Basket Analysis")

    try:
        rules = run_market_basket(df)

        unique_items = sorted(df["item_id"].unique())
        selected_item = st.selectbox("Select an Item:", unique_items)

        if st.button("Generate Recommendations"):
            recs = recommend_market_basket(
                item=selected_item,
                rules_df=rules,
                k=5
            )

            if recs:
                st.write("Associated Items:")
                st.table(pd.DataFrame({"Recommended Items": recs}))
            else:
                st.info("No associated items found for this selection.")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Developed by Abhishek Jha")
