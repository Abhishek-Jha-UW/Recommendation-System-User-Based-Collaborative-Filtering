import streamlit as st
import pandas as pd
from model import RecommenderEngine
import io

# --- Page Setup ---
st.set_page_config(page_title="AI Discovery Engine", layout="wide", page_icon="🚀")

# --- Default Sample Data (Internal) ---
def get_sample_data():
    data = {
        'user_id': ['User_A', 'User_A', 'User_B', 'User_C', 'User_C', 'User_D', 'User_E', 'User_E'],
        'item_name': ['Elden Ring', 'Halo', 'Elden Ring', 'FIFA 24', 'Halo', 'Elden Ring', 'FIFA 24', 'Zelda'],
        'rating': [5, 4, 5, 2, 5, 4, 5, 5]
    }
    return pd.DataFrame(data)

# --- UI Header ---
st.title("🚀 AI Product Recommender")
st.markdown("Upload your data or use our **Sample Mode** to see the engine in action.")

# --- Sidebar: Data Controls ---
with st.sidebar:
    st.header("1. Data Input")
    mode = st.radio("Choose Data Source:", ["Use Sample Data", "Upload My Own"])
    
    if mode == "Upload My Own":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Awaiting file...")
            st.stop()
    else:
        df = get_sample_data()
        st.success("Loaded internal sample dataset.")

    st.divider()
    st.header("2. Get Template")
    # Create template for user download
    template_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating'])
    csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV Template", data=csv, file_name="template.csv", mime="text/csv")

# --- Main Logic ---
engine = RecommenderEngine(df)
engine.build_similarity()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuration")
    selected_user = st.selectbox("Select a User", df.iloc[:, 0].unique())
    num_recs = st.number_input("Recommendations count", 1, 10, 3)
    
    with st.expander("View Training Data"):
        st.dataframe(df, use_container_width=True)

with col2:
    st.subheader("Your Personalized Recommendations")
    if st.button("Generate Predictions"):
        recs = engine.get_user_recommendations(selected_user, n=num_recs)
        
        if not recs.empty:
            # Create a nice display table
            results_df = pd.DataFrame({
                "Product/Item": recs.index,
                "Match Score": [f"{round(val * 20, 1)}%" for val in recs.values] 
            })
            st.table(results_df)
            st.balloons()
        else:
            st.warning("Not enough data for this specific user. Try a different user!")

# --- Footer ---
st.divider()
st.caption("Custom Engine built by Abhishek Jha | No External API required")
