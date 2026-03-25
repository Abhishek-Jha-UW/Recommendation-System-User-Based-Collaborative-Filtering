import streamlit as st
import pandas as pd
import urllib.request
import io
import model  # Ensure model.py is in the same folder

st.set_page_config(page_title="Game Recommender AI", layout="wide")

@st.cache_data
def load_data(uploaded_file, url):
    try:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            # Adding a timeout and specific error handling for the URL
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()
                df = pd.read_csv(io.BytesIO(data))
        
        # Take only the first 3 columns to avoid structure errors
        df = df.iloc[:, :3] 
        df.columns = ['user_id', 'item_id', 'rating']
        return df
    except urllib.error.HTTPError as e:
        st.error(f"**GitHub Error {e.code}:** The file was not found at the URL provided. Please verify the link below.")
        st.code(url)
        return None
    except Exception as e:
        st.error(f"**Loading Error:** {e}")
        return None

# --- Configuration ---
st.title("🎯 Pro Recommendation Engine")
st.sidebar.header("Data Settings")

# CHECK THIS URL: Ensure 'main' is the correct branch name for your repo
github_url = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df = load_data(uploaded_file, github_url)

if df is not None:
    # Build Pivot and Similarity
    with st.spinner("Processing Data..."):
        # Removing duplicates is critical for the .pivot() function to work
        df_clean = df.drop_duplicates(subset=['user_id', 'item_id'])
        pivot = df_clean.pivot(index='user_id', columns='item_id', values='rating')
        user_sim = model.compute_similarity(pivot, target="user")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("User Selection")
        user_id = st.selectbox("Select User ID:", pivot.index)
        num_recs = st.slider("Number of results:", 1, 10, 5)
        
        history = df[df['user_id'] == user_id]
        st.metric("Total User Ratings", len(history))
        with st.expander("View Rated Items"):
            st.dataframe(history[['item_id', 'rating']], use_container_width=True)

    with right_col:
        st.subheader("Top Recommendations")
        if st.button("Generate Recommendations"):
            with st.spinner("Calculating mathematical similarities..."):
                recs = model.get_recommendations(user_id, pivot, user_sim, n=num_recs)
                if recs:
                    rec_df = pd.DataFrame(recs, columns=['Product Name', 'Predicted Rating'])
                    # Visual styling for the table
                    st.dataframe(rec_df.style.format({"Predicted Rating": "{:.2f}"}), use_container_width=True)
                else:
                    st.info("No items left to recommend (User may have rated everything).")
else:
    st.info("Waiting for data... If the GitHub link is failing, try uploading the CSV file manually in the sidebar.")

st.divider()
st.caption("Developed by Abhishek Jha | Build 2026.03")
