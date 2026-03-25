import streamlit as st
import pandas as pd
import urllib.request
import io
import model  # Ensure model.py is in the same folder

# --- Page Config ---
st.set_page_config(page_title="Game Recommender AI", layout="wide")

@st.cache_data
def load_data(uploaded_file, url):
    """Loads data from upload or GitHub URL."""
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Revised Request block
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                # Read the bytes and decode
                data = response.read()
                df = pd.read_csv(io.BytesIO(data))
        
        # Mapping columns: We assume the first 3 columns are User, Item, Rating
        # This prevents errors if the CSV headers don't match exactly
        df = df.iloc[:, :3] 
        df.columns = ['user_id', 'item_id', 'rating']
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- Main Logic ---
st.title("🎯 Pro Recommendation Engine")
st.sidebar.header("Data Settings")

# IMPORTANT: Ensure this URL is exactly correct. 
# Check if 'games_sample.csv' is in the main branch or a different one.
github_url = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df = load_data(uploaded_file, github_url)

if df is not None:
    # Build Pivot and Similarity
    with st.spinner("Building Similarity Matrix..."):
        # We drop duplicates to ensure pivot_table doesn't error out
        df_clean = df.drop_duplicates(subset=['user_id', 'item_id'])
        pivot = df_clean.pivot(index='user_id', columns='item_id', values='rating')
        user_sim = model.compute_similarity(pivot, target="user")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("User Selection")
        user_id = st.selectbox("Pick a User:", pivot.index)
        num_recs = st.slider("Number of results:", 1, 10, 5)
        
        history = df[df['user_id'] == user_id]
        st.metric("Items Rated", len(history))
        with st.expander("Show Rating History"):
            st.dataframe(history[['item_id', 'rating']], use_container_width=True)

    with right_col:
        st.subheader("Top Picks for You")
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing preferences..."):
                recs = model.get_recommendations(user_id, pivot, user_sim, n=num_recs)
                if recs:
                    rec_df = pd.DataFrame(recs, columns=['Product Name', 'Predicted Rating'])
                    st.table(rec_df)
                else:
                    st.info("No recommendations found.")
else:
    st.warning("Could not reach the dataset. Please upload your own CSV file in the sidebar.")
