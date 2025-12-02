import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ----------------------------
# Cached helpers
# ----------------------------

@st.cache_data
def prepare_matrices(df):
    """
    Prepare sparse rating matrix and mappings from dataframe with columns:
    user_id, product_name, rating
    Returns a dict containing mappings and per-user stats.
    """
    df = df.copy()
    df['user_id'] = df['user_id'].astype(str)
    df['product_name'] = df['product_name'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    df = df.dropna(subset=['rating', 'user_id', 'product_name'])
    df = df.reset_index(drop=True)

    users = df['user_id'].unique().tolist()
    products = df['product_name'].unique().tolist()

    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    prod_to_idx = {p: i for i, p in enumerate(products)}
    idx_to_prod = {i: p for p, i in prod_to_idx.items()}

    user_idx = df['user_id'].map(user_to_idx).to_numpy()
    prod_idx = df['product_name'].map(prod_to_idx).to_numpy()
    ratings = df['rating'].to_numpy()

    n_users = len(users)
    n_prods = len(products)

    rating_csr = csr_matrix((ratings, (user_idx, prod_idx)), shape=(n_users, n_prods))

    user_counts = np.diff(rating_csr.indptr)
    user_sums = np.asarray(rating_csr.sum(axis=1)).ravel()
    user_means = np.zeros(n_users, dtype=float)
    non_zero_mask = user_counts > 0
    user_means[non_zero_mask] = user_sums[non_zero_mask] / user_counts[non_zero_mask]

    return {
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'prod_to_idx': prod_to_idx,
        'idx_to_prod': idx_to_prod,
        'rating_csr': rating_csr,
        'user_counts': user_counts,
        'user_sums': user_sums,
        'user_means': user_means
    }

@st.cache_resource
def build_nn_model(_rating_csr, algorithm='brute'):
    """Build and cache a NearestNeighbors model."""
    nn = NearestNeighbors(metric='cosine', algorithm=algorithm, n_jobs=-1)
    nn.fit(_rating_csr)
    return nn

# ----------------------------
# Prediction & recommendation logic
# ----------------------------

def predict_rating_user_based_sparse(user_idx, prod_idx, data_dict, nn_model, top_k=100, rating_min=1, rating_max=10):
    rating_csr = data_dict['rating_csr']
    user_means = data_dict['user_means']

    if data_dict['user_counts'][user_idx] == 0:
        return None

    users_who_rated = rating_csr[:, prod_idx].nonzero()
    if users_who_rated.size == 0:
        return None

    n_users = rating_csr.shape
    k_query = min(n_users, max(10, top_k + 10))
    
    # Robust neighborhood query
    try:
        distances, neighbor_indices = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=k_query, return_distance=True)
    except Exception:
        k_query = min(n_users, 10)
        distances, neighbor_indices = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=k_query, return_distance=True)

    distances = np.ravel(distances)
    neighbor_indices = np.ravel(neighbor_indices)
    neigh_sims = 1.0 - distances

    mask_rated = np.isin(neighbor_indices, users_who_rated)
    filtered_neighbors = neighbor_indices[mask_rated]
    filtered_sims = neigh_sims[mask_rated]

    if filtered_neighbors.size == 0:
        return None # Cannot find a suitable neighbor overlap

    valid_mask = ~np.isnan(filtered_sims) & (np.abs(filtered_sims) > 1e-8)
    filtered_neighbors = filtered_neighbors[valid_mask]
    filtered_sims = filtered_sims[valid_mask]

    if filtered_neighbors.size == 0:
        return None

    if filtered_neighbors.size > top_k:
        top_k_idx = np.argsort(filtered_sims)[-top_k:]
        filtered_neighbors = filtered_neighbors[top_k_idx]
        filtered_sims = filtered_sims[top_k_idx]

    neighbor_ratings = rating_csr[filtered_neighbors, prod_idx].toarray().ravel()
    neighbor_means = data_dict['user_means'][filtered_neighbors]
    ratings_diff = neighbor_ratings - neighbor_means

    numerator = np.dot(filtered_sims, ratings_diff)
    denominator = np.sum(np.abs(filtered_sims))
    if denominator == 0:
        return None

    user_mean = user_means[user_idx]
    prediction = user_mean + (numerator / denominator)
    prediction = float(np.clip(prediction, rating_min, rating_max))
    return prediction

def get_top_n_recommendations_sparse(user_idx, data_dict, nn_model, n=5, top_k_neighbors=100):
    rating_csr = data_dict['rating_csr']
    user_row = rating_csr[user_idx]
    unrated_mask = (user_row.toarray().ravel() == 0)
    unrated_prod_idx = np.where(unrated_mask).tolist()

    if not unrated_prod_idx:
        return []

    preds = {}
    for p_idx in unrated_prod_idx:
        pred = predict_rating_user_based_sparse(user_idx, p_idx, data_dict, nn_model, top_k=top_k_neighbors)
        if pred is not None:
            preds[p_idx] = pred

    top_items = sorted(preds.items(), key=lambda x: x, reverse=True)[:n]
    
    # Convert index predictions back to product names
    idx_to_prod = data_dict['idx_to_prod']
    named_recommendations = [(idx_to_prod[idx], rating) for idx, rating in top_items]
    return named_recommendations

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="CF Recommender (Robust)", layout="centered")
st.title("Collaborative Filtering Recommendation System")

st.markdown("""
This version loads data directly from a public GitHub repository using sparse matrices + KNN for better performance.
""")

st.markdown("---")

# Define the GitHub raw data URL
github_csv_url = "raw.githubusercontent.com"

# Load data automatically using pandas from the URL
try:
    with st.spinner(f"Loading data from {github_csv_url}..."):
        df = pd.read_csv(github_csv_url) # pandas can read directly from the raw URL

    if len(df.columns) != 3:
        st.error("Please ensure your file has exactly 3 columns (User ID, Product Name, Rating).")
        st.stop()

    df.columns = ['user_id', 'product_name', 'rating']
    st.success("Sample data loaded and processed from GitHub.")

    with st.spinner("Preparing sparse matrices and KNN model..."):
        data_dict = prepare_matrices(df)
        rating_csr = data_dict['rating_csr']
        
        n_users, n_products = rating_csr.shape

        if n_users < 2:
            st.error("Need at least 2 distinct users in the data to run collaborative filtering.")
            st.stop()

        nn_model = build_nn_model(rating_csr)

    st.write(f"Users: {n_users} — Products: {n_products} — Ratings (non-zero): {rating_csr.nnz}")

    # --- Generate Recommendations UI ---
    st.subheader("Generate Recommendations")
    
    user_list = list(data_dict['user_to_idx'].keys())
    target_user_id = st.selectbox(
        "Select a User ID to generate recommendations for:",
        options=user_list
    )
    
    if st.button("Get Top 5 Recommendations"):
        with st.spinner(f"Generating recommendations for User {target_user_id}..."):
            user_idx = data_dict['user_to_idx'][target_user_id]
            recommendations = get_top_n_recommendations_sparse(user_idx, data_dict, nn_model, n=5)
            
            if recommendations:
                st.write(f"### Top 5 Recommended Products for User {target_user_id}:")
                rec_df = pd.DataFrame(recommendations, columns=['Product Name', 'Predicted Rating'])
                st.table(rec_df)
            else:
                st.info(f"No new recommendations could be generated for User {target_user_id}.")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please check the GitHub URL and ensure the file format is correct.")



# --- 4. Add the Footer ---
st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
