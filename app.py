import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Helper Functions
# ----------------------------

@st.cache_data
def load_data(file_url=None, uploaded_file=None):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        df = pd.read_csv(file_url)
    df.columns = ['user_id', 'product_name', 'rating']
    return df

@st.cache_data
def create_rating_matrix(df):
    return df.pivot_table(index='user_id', columns='product_name', values='rating')

@st.cache_data
def compute_user_similarity(ratings_matrix):
    filled = ratings_matrix.fillna(0)
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=ratings_matrix.index, columns=ratings_matrix.index)

def predict_rating(user_id, product, ratings_matrix, user_sim):
    if product not in ratings_matrix.columns:
        return None
    
    # Users who rated this product
    rated_users = ratings_matrix[product].dropna()
    if rated_users.empty:
        return None
    
    # Similarities of target user with those users
    sims = user_sim.loc[user_id, rated_users.index]
    if sims.sum() == 0:
        return None
    
    # Weighted average deviation from user means
    user_mean = ratings_matrix.loc[user_id].mean()
    neighbors_mean = ratings_matrix.loc[rated_users.index].mean(axis=1)
    rating_diff = rated_users - neighbors_mean
    pred = user_mean + np.dot(sims, rating_diff) / sims.sum()
    return np.clip(pred, 1, 10)

def get_top_n_recommendations(user_id, ratings_matrix, user_sim, n=5):
    user_ratings = ratings_matrix.loc[user_id]
    unrated_products = user_ratings[user_ratings.isna()].index.tolist()
    
    preds = {}
    for product in unrated_products:
        rating_pred = predict_rating(user_id, product, ratings_matrix, user_sim)
        if rating_pred is not None:
            preds[product] = rating_pred
    top_n = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_n

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="CF Recommender", layout="centered")
st.title("User-Based Collaborative Filtering Recommender")

st.markdown("""
This app generates recommendations using **user-user collaborative filtering** with cosine similarity.  
It works **fast even on moderately large datasets**.
""")

# File selection
uploaded_file = st.file_uploader("Upload your CSV/Excel file (Optional)", type=["csv", "xlsx"])
github_csv_url = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"

df = load_data(file_url=github_csv_url, uploaded_file=uploaded_file)
st.success(f"Dataset loaded: {df.shape[0]} ratings, {df['user_id'].nunique()} users, {df['product_name'].nunique()} products.")

# Rating matrix & similarity
with st.spinner("Creating rating matrix and computing user similarity..."):
    ratings_matrix = create_rating_matrix(df)
    user_sim = compute_user_similarity(ratings_matrix)

# Generate recommendations
st.subheader("Generate Recommendations")
user_list = ratings_matrix.index.tolist()
target_user = st.selectbox("Select a User ID to generate recommendations for:", user_list)

if st.button("Get Top 5 Recommendations"):
    with st.spinner(f"Generating recommendations for User {target_user}..."):
        recommendations = get_top_n_recommendations(target_user, ratings_matrix, user_sim, n=5)
        if recommendations:
            rec_df = pd.DataFrame(recommendations, columns=['Product Name', 'Predicted Rating'])
            st.write(f"### Top 5 Recommendations for User {target_user}:")
            st.table(rec_df)
        else:
            st.info("No recommendations could be generated for this user.")

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
