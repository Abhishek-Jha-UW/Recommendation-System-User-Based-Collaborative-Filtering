import streamlit as st
import pandas as pd
import model  # Importing your engine logic

# --- Configuration ---
st.set_page_config(page_title="AI Recommender Pro", layout="wide")

@st.cache_data
def load_and_clean_data(file):
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    # Ensure standard naming for the engine
    df.columns = ['user_id', 'item_id', 'rating'] 
    return df

# --- Sidebar & File Loading ---
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])

# Use your GitHub URL as default if no file is uploaded
DATA_URL = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"
raw_data = load_and_clean_data(uploaded_file) if uploaded_file else pd.read_csv(DATA_URL)

# --- App Body ---
st.title("🎯 Recommendation Engine")
st.markdown("This system uses **User-User Collaborative Filtering** with mean-centering to predict what you'll love next.")

# 1. Processing
pivot = raw_data.pivot_table(index='user_id', columns='item_id', values='rating')
user_sim = model.compute_similarity(pivot, target="user")

# 2. UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Selection")
    target_user = st.selectbox("Select User ID", pivot.index)
    num_rec = st.slider("Number of recommendations", 1, 10, 5)
    
    user_history = raw_data[raw_data['user_id'] == target_user]
    st.metric("Total Ratings by User", len(user_history))
    with st.expander("View User History"):
        st.dataframe(user_history[['item_id', 'rating']], use_container_width=True)

with col2:
    st.subheader("Top Recommendations")
    if st.button(f"Generate for User {target_user}"):
        with st.spinner("Calculating preferences..."):
            recs = model.get_recommendations(target_user, pivot, user_sim, n=num_rec)
            
            if recs:
                rec_df = pd.DataFrame(recs, columns=['Product Name', 'Predicted Score'])
                st.table(rec_df)
                
                # Visual Feedback
                st.success(f"Successfully generated {len(recs)} predictions.")
            else:
                st.warning("Not enough data to generate unique recommendations for this user.")

st.divider()
st.caption("Developed by Abhishek Jha | Powered by Cosine Similarity & Mean-Centering")
