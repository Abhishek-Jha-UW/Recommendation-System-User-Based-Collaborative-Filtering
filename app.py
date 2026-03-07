import streamlit as st
import pandas as pd
from model import (
    compute_user_similarity,
    recommend_user_based,
    compute_item_similarity,
    recommend_item_based,
    run_market_basket,
    recommend_market_basket
)

st.set_page_config(page_title="Recommendation System", layout="wide")

st.title("Recommendation System")

st.markdown("""
Upload your dataset or use the sample dataset provided below.
Supported methods:
- User-Based Collaborative Filtering
- Item-Based Collaborative Filtering
- Market Basket Analysis
""")

# ---------------------------------------------------------
# Sample dataset
# ---------------------------------------------------------
sample_df = pd.DataFrame({
    "user_id": ["U1", "U1", "U2", "U2", "U3", "U3"],
    "item_id": ["A", "B", "A", "C", "B", "D"],
    "rating": [5, 3, 4, 2, 5, 4]
})

template_csv = """user_id,item_id,rating
U1,A,5
U1,B,3
U2,A,4
U2,C,2
U3,B,5
U3,D,4
"""

st.download_button(
    "Download Template CSV",
    data=template_csv,
    file_name="template.csv",
    mime="text/csv"
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset uploaded.")
else:
    df = sample_df.copy()
    st.info("Using sample dataset.")

st.write("Preview:")
st.dataframe(df.head())

# ---------------------------------------------------------
# Method selection
# ---------------------------------------------------------
method = st.selectbox(
    "Select Recommendation Method",
    ["User-Based Collaborative Filtering", "Item-Based Collaborative Filtering", "Market Basket Analysis"]
)

# ---------------------------------------------------------
# USER-BASED CF
# ---------------------------------------------------------
if method == "User-Based Collaborative Filtering":
    user_item, sim_df = compute_user_similarity(df)
    users = list(user_item.index)

    selected_user = st.selectbox("Select User", users)

    if st.button("Generate Recommendations"):
        recs = recommend_user_based(selected_user, user_item, sim_df, k=5)
        st.write("Recommendations:")
        st.table(pd.DataFrame({"Recommended Items": recs}))

# ---------------------------------------------------------
# ITEM-BASED CF
# ---------------------------------------------------------
elif method == "Item-Based Collaborative Filtering":
    user_item, sim_df = compute_item_similarity(df)
    users = list(user_item.index)

    selected_user = st.selectbox("Select User", users)

    if st.button("Generate Recommendations"):
        recs = recommend_item_based(selected_user, user_item, sim_df, k=5)
        st.write("Recommendations:")
        st.table(pd.DataFrame({"Recommended Items": recs}))

# ---------------------------------------------------------
# MARKET BASKET
# ---------------------------------------------------------
else:
    rules = run_market_basket(df)
    items = sorted(df["item_id"].unique())

    selected_item = st.selectbox("Select Item", items)

    if st.button("Generate Associated Items"):
        recs = recommend_market_basket(selected_item, rules, k=5)
        st.write("Associated Items:")
        st.table(pd.DataFrame({"Recommended Items": recs}))
