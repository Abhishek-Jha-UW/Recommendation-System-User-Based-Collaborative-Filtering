import streamlit as st
import pandas as pd
import model

st.set_page_config(page_title="Analytics | Recommendation Engine", layout="wide")

# --- Optimized Cache Layer ---
@st.cache_data(show_spinner="Performing matrix calculations...")
def get_engine_data(df, method):
    if method == "Market Basket Analysis":
        return model.run_market_basket(df), None
    target = 'user' if "User-Based" in method else 'item'
    return model.compute_similarity(df, target=target)

# --- UI Sidebar ---
with st.sidebar:
    st.title("Settings")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    method = st.selectbox("Algorithm", [
        "User-Based Collaborative Filtering", 
        "Item-Based Collaborative Filtering", 
        "Market Basket Analysis"
    ])
    
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        # Professional fallback dataset
        data = pd.DataFrame({
            "user_id": ["U1", "U1", "U2", "U2", "U3", "U3", "U4", "U4"],
            "item_id": ["Product_A", "Product_B", "Product_A", "Product_C", "Product_B", "Product_D", "Product_A", "Product_D"],
            "rating": [5, 4, 4, 2, 5, 4, 3, 5]
        })

# --- Main Interface ---
st.title("Predictive Recommendation Dashboard")
st.divider()

col_input, col_output = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("Configuration")
    engine_result, sim_df = get_engine_data(data, method)
    
    if "Collaborative" in method:
        user_list = engine_result.index.tolist()
        selected = st.selectbox("Select Target User", user_list)
        if st.button("Generate Predictions"):
            if "User-Based" in method:
                recs = model.recommend_user_based(selected, engine_result, sim_df)
            else:
                recs = model.recommend_item_based(selected, engine_result, sim_df)
            st.session_state['results'] = recs
    else:
        item_list = sorted(data["item_id"].unique())
        selected = st.selectbox("Select Anchor Item", item_list)
        if st.button("Run Association Rules"):
            recs = engine_result[engine_result["antecedent"] == selected]
            st.session_state['results'] = recs["consequent"].tolist()

with col_output:
    st.subheader("Output")
    if 'results' in st.session_state:
        if st.session_state['results']:
            st.table(pd.DataFrame(st.session_state['results'], columns=["Recommended Items"]))
        else:
            st.info("No strong associations found for this selection.")
    else:
        st.caption("Awaiting execution...")

    with st.expander("Data Preview"):
        st.dataframe(data.head(10), use_container_width=True)
