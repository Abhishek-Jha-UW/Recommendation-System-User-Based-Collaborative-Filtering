import streamlit as st
import pandas as pd
import urllib.request
import io
import model  # This imports your model.py file

# --- Page Config ---
st.set_page_config(page_title="Game Recommender AI", layout="wide")

@st.cache_data
def load_data(uploaded_file, url):
    """Loads data from upload or GitHub URL with custom headers."""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Bypass GitHub bot security with User-Agent header
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                df = pd.read_csv(io.BytesIO(response.read()))
        
        # Standardize column names for the model
        # Assumes format: User, Item, Rating
        df.columns = ['user_id', 'item_id', 'rating']
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- Main Logic ---
st.title("🎯 Pro Recommendation Engine")
st.sidebar.header("Data Settings")

github_url = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df = load_data(uploaded_file, github_url)

if df is not None:
    # 1. Prepare Data
    with st.spinner("Building Similarity Matrix..."):
        pivot = df.pivot_table(index='user_id', columns='item_id', values='rating')
        user_sim = model.compute_similarity(pivot, target="user")

    # 2. UI Layout
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("User Selection")
        user_id = st.selectbox("Pick a User to Recommend For:", pivot.index)
        num_recs = st.slider("Number of results:", 1, 10, 5)
        
        # Show user stats
        history = df[df['user_id'] == user_id]
        st.metric("Items Rated", len(history))
        with st.expander("Show Rating History"):
            st.dataframe(history[['item_id', 'rating']], use_container_width=True)

    with right_col:
        st.subheader("Top Picks for You")
        if st.button(f"Generate Recommendations"):
            with st.spinner("Analyzing neighbor preferences..."):
                recs = model.get_recommendations(user_id, pivot, user_sim, n=num_recs)
                
                if recs:
                    rec_df = pd.DataFrame(recs, columns=['Product Name', 'Predicted Rating'])
                    st.table(rec_df)
                    st.success("Calculated based on similar user behavior.")
                else:
                    st.info("No new items found to recommend for this user.")

else:
    st.warning("Please upload a dataset or check your internet connection to load the default data.")

st.markdown("---")
st.caption("Developed by Abhishek Jha | Collaborative Filtering Engine v2.0")
