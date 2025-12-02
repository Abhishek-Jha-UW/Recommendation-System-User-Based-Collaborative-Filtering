# cf_recommender_dense.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO

# ----------------------------
# Config
# ----------------------------
GITHUB_RAW_SAMPLE = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data
def load_sample_csv(url):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))

@st.cache_data
def prepare_ratings_matrix(df):
    df = df.copy()
    df.columns = ['user_id', 'product_name', 'rating']
    df['user_id'] = df['user_id'].astype(str)
    df['product_name'] = df['product_name'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating', 'user_id', 'product_name'])
    
    ratings_matrix = df.pivot_table(index='user_id', columns='product_name', values='rating')
    
    # Compute user-user cosine similarity
    similarity_matrix = pd.DataFrame(
        cosine_similarity(ratings_matrix.fillna(0)),
        index=ratings_matrix.index,
        columns=ratings_matrix.index
    )
    
    return ratings_matrix, similarity_matrix

def predict_rating_user_based(user_id, product_name, ratings_matrix, user_similarity_df):
    if product_name not in ratings_matrix.columns or user_id not in ratings_matrix.index:
        return None
    
    game_ratings = ratings_matrix[product_name]
    rated_users = game_ratings[game_ratings.notna()].index
    if len(rated_users) == 0:
        return None
    
    similarities = user_similarity_df.loc[user_id, rated_users]
    if similarities.sum() == 0 or similarities.isnull().all():
        return None
    
    neighbors_avg = ratings_matrix.loc[rated_users].mean(axis=1)
    ratings_diff = game_ratings[rated_users] - neighbors_avg
    
    numerator = (similarities * ratings_diff).sum()
    denominator = similarities.abs().sum()
    
    user_avg = ratings_matrix.loc[user_id].mean()
    prediction = user_avg + numerator / denominator
    
    # Clip to min/max observed ratings
    rmin = ratings_matrix.min().min()
    rmax = ratings_matrix.max().max()
    return np.clip(prediction, rmin, rmax)

def get_top_n_recommendations(user_id, ratings_matrix, user_similarity_df, n=5):
    user_ratings = ratings_matrix.loc[user_id]
    unrated_products = user_ratings[user_ratings.isna()].index.tolist()
    if not unrated_products:
        return []
    
    predictions = {}
    for product in unrated_products:
        pred = predict_rating_user_based(user_id, product, ratings_matrix, user_similarity_df)
        if pred is not None:
            predictions[product] = pred
    
    top_recommendations = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_recommendations

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="User-Based CF Recommender", layout="centered")
st.title("Collaborative Filtering — User-Based (Dense)")

st.markdown("Default sample CSV loaded from GitHub, or you can upload your own CSV/XLSX file.")

# Load GitHub sample CSV
with st.spinner("Loading sample CSV from GitHub..."):
    try:
        df = load_sample_csv(GITHUB_RAW_SAMPLE)
        st.success("Sample CSV loaded from GitHub.")
    except Exception as e:
        st.error(f"Failed to load sample CSV: {e}")
        st.stop()

# Optional file upload
uploaded_file = st.file_uploader("Or upload your own CSV/XLSX file:", type=["csv","xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.success("Uploaded file loaded.")

# Validate
if len(df.columns) != 3:
    st.error("CSV must have exactly 3 columns: User ID, Product Name, Rating")
    st.stop()

# Prepare ratings matrix and similarity
with st.spinner("Preparing ratings matrix and similarity matrix..."):
    ratings_matrix, user_similarity_df = prepare_ratings_matrix(df)

st.write(f"Users: {ratings_matrix.shape[0]} — Products: {ratings_matrix.shape[1]}")

# Top-N parameter
top_n = st.number_input("Top-N recommendations", min_value=1, max_value=50, value=5)

# Select user
target_user = st.selectbox("Select User ID:", ratings_matrix.index.tolist())

if st.button("Get Top Recommendations"):
    with st.spinner("Generating recommendations..."):
        recs = get_top_n_recommendations(target_user, ratings_matrix, user_similarity_df, n=top_n)
        if recs:
            rec_df = pd.DataFrame(recs, columns=['Product Name','Predicted Rating'])
            st.write(f"### Top {top_n} Recommendations for User {target_user}")
            st.table(rec_df)
        else:
            st.info("No recommendations could be generated for this user.")

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
