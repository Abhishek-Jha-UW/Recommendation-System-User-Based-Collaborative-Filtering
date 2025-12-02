# cf_recommender_streamlit_robust.py
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

    # Drop rows with missing rating or blanks
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

    # Per-user stats: counts, sums, means (safe conversions)
    user_counts = np.diff(rating_csr.indptr)  # non-zero count per user (row)
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
    """
    Build and cache a NearestNeighbors model.
    The leading underscore in the parameter prevents Streamlit from trying to hash the CSR argument.
    """
    nn = NearestNeighbors(metric='cosine', algorithm=algorithm, n_jobs=-1)
    nn.fit(_rating_csr)
    return nn

# ----------------------------
# Prediction & recommendation logic
# ----------------------------

def predict_rating_user_based_sparse(user_idx, prod_idx, data_dict, nn_model, top_k=100, rating_min=1, rating_max=10):
    """
    Predict rating for user_idx on prod_idx using neighbors found via nn_model.
    """
    rating_csr = data_dict['rating_csr']
    user_means = data_dict['user_means']

    # If target user has no ratings, cannot compute mean-adjusted prediction
    if data_dict['user_counts'][user_idx] == 0:
        return None

    # Users who rated the product
    users_who_rated = rating_csr[:, prod_idx].nonzero()[0]
    if users_who_rated.size == 0:
        return None

    n_users = rating_csr.shape[0]
    # Query nearest neighbors for the target user
    k_query = min(n_users, max(10, top_k + 10))
    try:
        distances, neighbor_indices = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=k_query, return_distance=True)
    except Exception:
        # fallback: ask for fewer neighbors if kneighbors fails
        k_query = min(n_users, 10)
        distances, neighbor_indices = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=k_query, return_distance=True)

    distances = np.ravel(distances)
    neighbor_indices = np.ravel(neighbor_indices)
    neigh_sims = 1.0 - distances  # convert cosine distance to similarity

    # Filter neighbors to those who rated the product
    mask_rated = np.isin(neighbor_indices, users_who_rated)
    filtered_neighbors = neighbor_indices[mask_rated]
    filtered_sims = neigh_sims[mask_rated]

    # Fallback: if none of the close neighbors rated the item, compute similarities directly vs users_who_rated
    if filtered_neighbors.size == 0:
        target_vec = rating_csr[user_idx]  # 1 x n_products sparse
        rated_matrix = rating_csr[users_who_rated]  # m x n_products sparse (m small)
        # dot products
        dots = rated_matrix.dot(target_vec.T).toarray().ravel()
        target_norm = np.sqrt(target_vec.multiply(target_vec).sum())
        rated_norms = np.sqrt(rated_matrix.multiply(rated_matrix).sum(axis=1))
        rated_norms = np.asarray(rated_norms).ravel()
        sims = np.zeros_like(dots, dtype=float)
        valid = (target_norm > 0) & (rated_norms > 0)
        if target_norm > 0:
            sims[valid] = dots[valid] / (target_norm * rated_norms[valid])
        filtered_neighbors = users_who_rated
        filtered_sims = sims

    # Remove invalid sims
    valid_mask = ~np.isnan(filtered_sims) & (np.abs(filtered_sims) > 1e-8)
    filtered_neighbors = filtered_neighbors[valid_mask]
    filtered_sims = filtered_sims[valid_mask]

    if filtered_neighbors.size == 0:
        return None

    # Limit to top_k neighbors by similarity
    if filtered_neighbors.size > top_k:
        top_k_idx = np.argsort(filtered_sims)[-top_k:]
        filtered_neighbors = filtered_neighbors[top_k_idx]
        filtered_sims = filtered_sims[top_k_idx]

    # Get neighbor ratings for the product
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
    # In this sparse representation, 0 means "no rating" (assuming ratings are >= 1)
    unrated_mask = (user_row.toarray().ravel() == 0)
    unrated_prod_idx = np.where(unrated_mask)[0].tolist()

    if not unrated_prod_idx:
        return []

    preds = {}
    # Option: to speed up, you can sample top popular products first; here we do all unrated
    for p_idx in unrated_prod_idx:
        pred = predict_rating_user_based_sparse(user_idx, p_idx, data_dict, nn_model, top_k=top_k_neighbors)
        if pred is not None:
            preds[p_idx] = pred

    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_items

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="CF Recommender (Robust)", layout="centered")
st.title("Collaborative Filtering Recommendation System (User_Based)")

st.markdown("""
Upload a CSV or Excel file with exactly three columns: **User ID, Product Name, Rating**.  
This version uses sparse matrices + KNN to avoid high memory usage.
""")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')

        if len(df.columns) != 3:
            st.error("Please ensure your file has exactly 3 columns (User ID, Product Name, Rating).")
            st.stop()

        df.columns = ['user_id', 'product_name', 'rating']
        st.success("File loaded.")

        with st.spinner("Preparing sparse matrices..."):
            data_dict = prepare_matrices(df)

        rating_csr = data_dict['rating_csr']
        n_users, n_products = rating_csr.shape
        st.write(f"Users: {n_users} — Products: {n_products} — Ratings (non-zero): {rating_csr.nnz}")

        if n_users < 2:
            st.error("Need at least 2 distinct users to run collaborative filtering.")
            st.stop()

        with st.spinner("Building neighbor-search index..."):
            nn_model = build_nn_model(rating_csr)

        st.subheader("Generate Recommendations")
        user_options = list(data_dict['user_to_idx'].keys())
        target_user = st.selectbox("Select a User ID to generate recommendations for:", options=user_options)

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
