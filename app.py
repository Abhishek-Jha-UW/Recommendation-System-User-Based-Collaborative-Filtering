import streamlit as st
import pandas as pd
from model import (
    compute_similarity,  # Using the optimized combined function
    recommend_user_based,
    recommend_item_based,
    run_market_basket,
    recommend_market_basket
)

# Page configuration for a clean, professional look
st.set_page_config(
    page_title="Recommendation Engine | Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for a more formal aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Persistence & Caching
# ---------------------------------------------------------
@st.cache_data
def load_and_compute(df, method):
    """Caches the heavy computation to prevent UI lag."""
    if method == "User-Based Collaborative Filtering":
        return compute_similarity(df, target='user')
    elif method == "Item-Based Collaborative Filtering":
        return compute_similarity(df, target='item')
    else:
        return run_market_basket(df), None

# ---------------------------------------------------------
# Sidebar - Data Management
# ---------------------------------------------------------
with st.sidebar:
    st.header("Data Configuration")
    uploaded = st.file_uploader("Upload Transactional CSV", type=["csv"])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Dataset active.")
    else:
        # Simplified sample data for professional presentation
        df = pd.DataFrame({
            "user_id": ["U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4"],
            "item_id": ["Alpha", "Beta", "Alpha", "Gamma", "Beta", "Delta", "Alpha", "Delta"],
            "rating": [5, 3, 4, 2, 5, 4, 3, 5]
        })
        st.info("Operating on system defaults.")

    st.divider()
    method = st.selectbox(
        "Analysis Methodology",
        ["User-Based Collaborative Filtering", 
         "Item-Based Collaborative Filtering", 
         "Market Basket Analysis"]
    )

# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------
st.title("Predictive Recommendation Engine")
st.caption("Advanced filtering and association rules for transactional data.")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Parameters")
    
    if "Collaborative" in method:
        user_item, sim_df = load_and_compute(df, method)
        target_id = st.selectbox("Select Target User ID", user_item.index)
        top_k = st.slider("Number of Recommendations", 1, 10, 5)
        
        if st.button("Execute Inference"):
            with st.spinner("Calculating scores..."):
                if "User-Based" in method:
                    recs = recommend_user_based(target_id, user_item, sim_df, k=top_k)
                else:
                    recs = recommend_item_based(target_id, user_item, sim_df, k=top_k)
                st.session_state['recs'] = recs
    
    else:
        rules, _ = load_and_compute(df, method)
        items = sorted(df["item_id"].unique())
        target_id = st.selectbox("Select Anchor Item", items)
        
        if st.button("Analyze Associations"):
            recs = recommend_market_basket(target_id, rules, k=5)
            st.session_state['recs'] = recs

with col2:
    st.subheader("Results")
    if 'recs' in st.session_state:
        if st.session_state['recs']:
            res_df = pd.DataFrame(st.session_state['recs'], columns=["Recommended Item ID"])
            st.table(res_df)
        else:
            st.warning("Insufficient data to generate a high-confidence recommendation for this selection.")
    else:
        st.info("Adjust parameters and execute inference to view results.")

    with st.expander("View Source Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)
