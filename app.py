# cf_recommender_streamlit_safe.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import math

# ----------------------------
# Helper / cached functions
# ----------------------------

@st.cache_data
def prepare_matrices(df):
    """
    Prepare mappings and sparse rating matrix from dataframe with columns:
    user_id, product_name, rating
    Returns:
        user_to_idx, idx_to_user, prod_to_idx, idx_to_prod,
        rating_csr (n_users x n_products), user_counts, user_sums, user_means
    """
    # Ensure correct dtypes
    df = df.copy()
    df['user_id'] = df['user_id'].astype(str)
    df['product_name'] = df['product_name'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Build index mappings
    users = df['user_id'].unique().tolist()
    products = df['product_name'].unique().tolist()
    user_to_idx = {u: i for i, u in enumerate(users)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    prod_to_idx = {p: i for i, p in enumerate(products)}
    idx_to_prod = {i: p for p, i in prod_to_idx.items()}

    # Convert to indices
    user_idx = df['user_id'].map(user_to_idx).to_numpy()
    prod_idx = df['product_name'].map(prod_to_idx).to_numpy()
    ratings = df['rating'].to_numpy()

    # Create sparse matrix (rows: users, cols: products)
    n_users = len(users)
    n_prods = len(products)
    rating_csr = csr_matrix((ratings, (user_idx, prod_idx)), shape=(n_users, n_prods))

    # Per-user stats (only count non-zero entries)
    user_counts = np.diff(rating_csr.indptr)  # number of non-zero entries per row
    user_sums = rating_csr.sum(axis=1).A1  # sum per user
    # Avoid division by zero
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
    """
    Build and cache NearestNeighbors model on sparse ratings matrix.
    Using 'brute' algorithm + cosine metric works well with sparse csr matrices.
    The leading underscore in the parameter name tells Streamlit not to attempt to hash it.
    """
    # Use cosine distance (1 - cosine_similarity)
    nn = NearestNeighbors(metric='cosine', algorithm=algorithm, n_jobs=-1)
    nn.fit(_rating_csr)
    return nn

# ----------------------------
# Prediction & recommendation logic
# ----------------------------

def predict_rating_user_based_sparse(user_idx, prod_idx, data_dict, nn_model, top_k=100, rating_min=1, rating_max=10):
    """
    Predict rating for user_idx on prod_idx using a nearest-neighbor strategy.
    - user_idx: integer index for target user
    - prod_idx: integer index for target product
    - data_dict: output of prepare_matrices
    - nn_model: fitted NearestNeighbors on rating_csr
    - top_k: how many neighbors to consider (small constant; e.g., 50 or 100)
    """
    rating_csr = data_dict['rating_csr']
    user_means = data_dict['user_means']

    n_users = rating_csr.shape[0]
    # If target user has no ratings, we cannot compute a mean-adjusted prediction
    if data_dict['user_counts'][user_idx] == 0:
        return None

    # Users who rated the product
    users_who_rated = rating_csr[:, prod_idx].nonzero()[0]
    if users_who_rated.size == 0:
        return None

    # Query nearest neighbors for the target user (we'll get distances; convert to similarity)
    # Request slightly more neighbors to increase chances some of them rated the item
    k_query = min(n_users, max(10, top_k + 5))
    distances, neighbor_indices = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=k_query, return_distance=True)
    distances = np.ravel(distances)
    neighbor_indices = np.ravel(neighbor_indices)

    # Convert distances to similarities (cosine distance -> similarity = 1 - distance)
    neigh_sims = 1.0 - distances

    # Filter neighbors to those who rated the product
    # We keep order and corresponding similarity
    mask_rated = np.isin(neighbor_indices, users_who_rated)
    filtered_neighbors = neighbor_indices[mask_rated]
    filtered_sims = neigh_sims[mask_rated]

    # If no neighbor among top_k rated the item, compute direct similarities only against users_who_rated (fallback)
    if filtered_neighbors.size == 0:
        # Compute similarity between target user vector and the users_who_rated set using sparse vector dot
        # similarity = dot(u, v) / (||u|| * ||v||) but since we used cosine distance before, this matches that notion.
        # We'll compute norms:
        target_vec = rating_csr[user_idx]
        # compute dot product with all users_who_rated at once:
        rated_matrix = rating_csr[users_who_rated]  # small slice
        dots = rated_matrix.dot(target_vec.T).A1  # dot products
        # compute norms
        # avoid zero norm users
        target_norm = np.sqrt(target_vec.multiply(target_vec).sum())
        rated_norms = np.sqrt(rated_matrix.multiply(rated_matrix).sum(axis=1)).A1
        with np.errstate(divide='ignore', invalid='ignore'):
            sims = np.zeros_like(dots, dtype=float)
            valid = (target_norm > 0) & (rated_norms > 0)
            if target_norm > 0:
                sims[valid] = dots[valid] / (target_norm * rated_norms[valid])
        filtered_neighbors = users_who_rated
        filtered_sims = sims

    # Remove NaNs and near-zero similarities
    valid_mask = ~np.isnan(filtered_sims) & (np.abs(filtered_sims) > 1e-8)
    filtered_neighbors = filtered_neighbors[valid_mask]
    filtered_sims = filtered_sims[valid_mask]

    if filtered_neighbors.size == 0:
        return None

    # Optionally limit to top_k most similar neighbors
    if filtered_neighbors.size > top_k:
        top_k_idx = np.argsort(filtered_sims)[-top_k:]
        filtered_neighbors = filtered_neighbors[top_k_idx]
        filtered_sims = filtered_sims[top_k_idx]

    # Fetch neighbor ratings for the product
    # rating_csr[neighbor, prod_idx] -> returns a tiny dense array
    neighbor_ratings = rating_csr[filtered_neighbors, prod_idx].A1
    neighbor_means = data_dict['user_means'][filtered_neighbors]

    # deviations (neighbor rating - neighbor mean)
    ratings_diff = neighbor_ratings - neighbor_means

    numerator = np.dot(filtered_sims, ratings_diff)
    denominator = np.sum(np.abs(filtered_sims))

    if denominator == 0:
        return None

    # target user mean
    user_mean = user_means[user_idx]
    prediction = user_mean + (numerator / denominator)

    # clamp
    prediction = float(np.clip(prediction, rating_min, rating_max))
    return prediction

def get_top_n_recommendations_sparse(user_idx, data_dict, nn_model, n=5, top_k_neighbors=100):
    rating_csr = data_dict['rating_csr']
    # Products not rated by user
    user_row = rating_csr[user_idx]
    unrated_mask = (user_row.toarray().ravel() == 0)  # zero indicates no explicit rating in this representation
    # but note: if an actual rating of 0 exists in your dataset you should use sentinel. Here ratings assumed >=1.
    unrated_prod_idx = np.where(unrated_mask)[0].tolist()

    if not unrated_prod_idx:
        return []

    preds = {}
    for p_idx in unrated_prod_idx:
        pred = predict_rating_user_based_sparse(user_idx, p_idx, data_dict, nn_model, top_k=top_k_neighbors)
        if pred is not None:
            preds[p_idx] = pred

    # sort and return top n as (product_index, predicted_rating)
    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_items

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="CF Recommender (Robust)", layout="centered")
st.title("Collaborative Filtering Recommendation System (Robust for large data)")

st.markdown("""
Upload your data file (CSV or Excel) with **three columns**: **User ID, Product Name, Rating**.  
This version uses sparse matrices and neighbor search to avoid high memory use.
""")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        if len(df.columns) != 3:
            st.error("Please ensure your file has exactly 3 columns (User ID, Product Name, Rating).")
            st.stop()

        df.columns = ['user_id', 'product_name', 'rating']
        st.success("File loaded.")

        # Prepare matrices (cached)
        with st.spinner("Preparing data (sparse matrices)..."):
            data_dict = prepare_matrices(df)

        rating_csr = data_dict['rating_csr']
        n_users, n_products = rating_csr.shape
        st.write(f"Users: {n_users} — Products: {n_products} — Ratings (non-zero): {rating_csr.nnz}")

        if n_users < 2:
            st.error("Need at least 2 distinct users to run collaborative filtering.")
            st.stop()

        # Build or load NN model (cached)
        with st.spinner("Building neighbor-search index..."):
            nn_model = build_nn_model(rating_csr)

        st.subheader("Generate Recommendations")
        # Show dropdown of user IDs (string)
        user_options = list(data_dict['user_to_idx'].keys())
        target_user = st.selectbox("Select a User ID to generate recommendations for:", options=user_options)

        # parameters
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.number_input("Top N recommendations", min_value=1, max_value=50, value=5)
        with col2:
            top_k_neighbors = st.number_input("Top-K neighbors to consider", min_value=5, max_value=500, value=100)

        if st.button("Get Recommendations"):
            user_idx = data_dict['user_to_idx'][str(target_user)]
            with st.spinner("Generating recommendations..."):
                recs = get_top_n_recommendations_sparse(user_idx, data_dict, nn_model, n=top_n, top_k_neighbors=top_k_neighbors)

            if recs:
                # Convert to readable DF
                rows = []
                for pid, pred in recs:
                    prod_name = data_dict['idx_to_prod'][pid]
                    rows.append({'Product Name': prod_name, 'Predicted Rating': round(pred, 3)})
                rec_df = pd.DataFrame(rows)
                st.write(f"### Top {top_n} recommendations for user {target_user}:")
                st.table(rec_df)
            else:
                st.info("No recommendations could be generated for this user. (Possibly user rated nothing, or no overlapping neighbors found.)")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
