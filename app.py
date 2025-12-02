# cf_recommender_streamlit_final.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests
from io import StringIO, BytesIO

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
    """Build and cache a NearestNeighbors model. Leading underscore avoids hashing the csr."""
    nn = NearestNeighbors(metric='cosine', algorithm=algorithm, n_jobs=-1)
    nn.fit(_rating_csr)
    return nn

# ----------------------------
# Prediction & recommendation (robust + vectorized)
# ----------------------------

def get_recommendations_user_based(user_idx,
                                   data_dict,
                                   nn_model,
                                   top_n=5,
                                   k_neighbors=50,
                                   similarity_threshold=0.0,
                                   max_k_expand=300):
    """
    Robust fast user-based CF:
      1) Query K nearest neighbors.
      2) Compute mean-adjusted weighted deviation scores for all items in one vectorized step.
      3) Remove items user already rated.
      4) Return top_n (product_name, predicted_score).

    Reliability features:
      - Uses neighbors' deviations from their own means (reduces bias).
      - Applies similarity threshold (set >0 for stricter neighbors).
      - If too few neighbors rated many items, will expand k up to max_k_expand.
    """
    rating_csr = data_dict['rating_csr']
    user_means = data_dict['user_means']

    n_users, n_products = rating_csr.shape

    # helper to run kneighbors safely
    def kneigh(kq):
        kq = min(max(2, int(kq)), n_users)
        distances, neighbors = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=kq, return_distance=True)
        return np.ravel(distances), np.ravel(neighbors)

    # initial neighbors
    k_query = min(k_neighbors + 1, n_users)
    distances, neighbors = kneigh(k_query)

    # drop self if present
    mask_self = neighbors != user_idx
    neighbors = neighbors[mask_self]
    distances = distances[mask_self]

    if neighbors.size == 0:
        return []

    sims = 1.0 - distances
    # apply similarity threshold (zero or above)
    good_mask = sims >= similarity_threshold
    neighbors = neighbors[good_mask]
    sims = sims[good_mask]

    # If we filtered too many neighbors (none left), fall back to top ones (no threshold)
    if neighbors.size == 0:
        distances, neighbors = kneigh(k_query)
        mask_self = neighbors != user_idx
        neighbors = neighbors[mask_self]
        distances = distances[mask_self]
        sims = 1.0 - distances

    # If still empty return
    if neighbors.size == 0:
        return []

    # If neighbors>K requested, trim to requested
    if neighbors.size > k_neighbors:
        sorted_idx = np.argsort(sims)[-k_neighbors:]
        neighbors = neighbors[sorted_idx]
        sims = sims[sorted_idx]

    # Now compute vectorized mean-adjusted predictions:
    # neighbor_ratings: shape (K, P) sparse
    neighbor_ratings = rating_csr[neighbors]            # sparse K x P
    # convert to small dense since K is intended to be small (e.g., 50-200)
    # This will be memory efficient as long as K is modest.
    neighbor_dense = neighbor_ratings.toarray()         # K x P (numpy)
    neighbor_means = data_dict['user_means'][neighbors] # K

    # mean-adjusted deviations
    deviations = neighbor_dense - neighbor_means.reshape(-1, 1)  # K x P

    # apply non-negative similarities (neg sims -> 0)
    sims = np.maximum(sims, 0.0)
    weights = sims.reshape(-1, 1)  # K x 1

    # Weighted sum of deviations across neighbors
    weighted_dev = (deviations * weights).sum(axis=0)  # vector length P

    # Denominator: sum of absolute weights for users who actually rated the item
    # Build mask where neighbor actually rated the item
    rated_mask_by_neighbor = neighbor_dense != 0  # K x P boolean
    abs_weights = np.abs(weights)
    denom = (rated_mask_by_neighbor * abs_weights).sum(axis=0)  # vector length P

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pred_dev = np.zeros_like(weighted_dev, dtype=float)
        nonzero = denom > 0
        pred_dev[nonzero] = weighted_dev[nonzero] / denom[nonzero]

    # Final predicted rating = user_mean + predicted deviation
    target_user_mean = data_dict['user_means'][user_idx]
    preds = target_user_mean + pred_dev  # vector length P

    # If too many items have no denom (i.e., neighbors didn't rate them),
    # expand neighbors progressively up to max_k_expand to try get coverage.
    if np.sum(denom > 0) < max(10, top_n * 3) and len(neighbors) < max_k_expand:
        # try expanding (double k) but limited
        new_k = min(max(len(neighbors) * 2, k_neighbors * 2), max_k_expand)
        if new_k > len(neighbors):
            distances2, neighbors2 = kneigh(new_k + 1)
            mask_self2 = neighbors2 != user_idx
            neighbors2 = neighbors2[mask_self2]
            distances2 = distances2[mask_self2]
            sims2 = 1.0 - distances2
            # take top new_k by similarity (no threshold in expand)
            sorted_idx2 = np.argsort(sims2)[-new_k:]
            neighbors2 = neighbors2[sorted_idx2]
            sims2 = sims2[sorted_idx2]

            # recompute using expanded neighbors (same logic as above)
            neighbor_dense = rating_csr[neighbors2].toarray()
            neighbor_means = data_dict['user_means'][neighbors2]
            deviations = neighbor_dense - neighbor_means.reshape(-1, 1)
            sims2 = np.maximum(sims2, 0.0)
            weights2 = sims2.reshape(-1, 1)
            weighted_dev = (deviations * weights2).sum(axis=0)
            rated_mask_by_neighbor = neighbor_dense != 0
            denom = (rated_mask_by_neighbor * np.abs(weights2)).sum(axis=0)
            pred_dev = np.zeros_like(weighted_dev, dtype=float)
            nonzero = denom > 0
            pred_dev[nonzero] = weighted_dev[nonzero] / denom[nonzero]
            preds = target_user_mean + pred_dev

    # Clip predictions to a reasonable range (assume original ratings were within observed min/max)
    # infer range from data if possible
    all_obs = data_dict['rating_csr'].data
    if all_obs.size > 0:
        rmin, rmax = float(all_obs.min()), float(all_obs.max())
    else:
        rmin, rmax = 1.0, 10.0
    preds = np.clip(preds, rmin, rmax)

    # remove items already rated by the target user
    user_row = data_dict['rating_csr'][user_idx].toarray().ravel()
    already_rated = user_row != 0
    preds[already_rated] = -np.inf

    # pick top_n
    top_idx = np.argsort(preds)[-top_n:][::-1]
    top_idx = [int(i) for i in top_idx if preds[i] != -np.inf]

    idx_to_prod = data_dict['idx_to_prod']
    recs = [(idx_to_prod[i], float(round(preds[i], 4))) for i in top_idx]
    return recs

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="CF Recommender (Robust v2)", layout="centered")
st.title("Collaborative Filtering — Robust User-Based (Adaptive K)")

st.markdown("""
Upload a CSV or Excel file with exactly three columns: **User ID, Product Name, Rating**.  
This app uses sparse matrices + KNN and an adaptive, mean-adjusted scoring to provide accurate and fast user-based recommendations.
""")

st.markdown("---")
st.subheader("Load sample CSV from GitHub (optional)")
st.markdown("https://github.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/blob/main/games_sample.csv")

github_raw_url = st.text_input("Paste raw.githubusercontent.com URL for sample CSV (optional):",
                              value="https://raw.githubusercontent.com/<username>/<repo>/main/games_sample.csv")

col_s1, col_s2 = st.columns([1, 3])
with col_s1:
    if st.button("Load sample from GitHub"):
        if "<username>" in github_raw_url or "raw.githubusercontent.com" not in github_raw_url:
            st.warning("Please replace the placeholder with your actual raw.githubusercontent.com URL before clicking 'Load sample'.")
        else:
            try:
                with st.spinner("Downloading sample CSV..."):
                    resp = requests.get(github_raw_url, timeout=15)
                    resp.raise_for_status()
                    sample_df = pd.read_csv(StringIO(resp.text))
                    st.session_state["_sample_df"] = sample_df
                    st.success("Sample CSV downloaded and loaded.")
            except Exception as e:
                st.error(f"Could not download sample CSV: {e}")

with col_s2:
    st.write("Or upload your own CSV/Excel below.")

st.markdown("---")

uploaded_file = st.file_uploader("Choose a CSV or Excel file (or load the GitHub sample above)", type=["csv", "xlsx"])

# If user loaded sample into session_state
if " _sample_df" in st.session_state and st.session_state["_sample_df"] is not None:
    # ignore; this key name has a leading space to avoid accidental collisions (rare)
    pass

# prefer uploaded file; else sample in session
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")
        st.stop()
else:
    # check session state for the sample key we used earlier
    if st.session_state.get("_sample_df") is not None:
        df = st.session_state["_sample_df"]

if df is None:
    st.info("Upload a CSV/XLSX file or load a sample from GitHub to proceed.")
    st.stop()

# Validate structure
if len(df.columns) != 3:
    st.error("Please ensure your file has exactly 3 columns: User ID, Product Name, Rating.")
    st.stop()

df.columns = ['user_id', 'product_name', 'rating']
st.success("File ready — preparing matrices...")

try:
    with st.spinner("Preparing data and building KNN index..."):
        data_dict = prepare_matrices(df)
        rating_csr = data_dict['rating_csr']
        n_users, n_products = rating_csr.shape

        if n_users < 2:
            st.error("Need at least 2 distinct users to run collaborative filtering.")
            st.stop()

        nn_model = build_nn_model(rating_csr)

    st.write(f"Users: {n_users} — Products: {n_products} — Ratings (non-zero): {rating_csr.nnz}")

    # UI for recommendation parameters
    st.subheader("Recommendation parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.number_input("Top-N results", min_value=1, max_value=50, value=5)
    with col2:
        k_neighbors = st.number_input("Initial K neighbors", min_value=5, max_value=500, value=50)
    with col3:
        sim_thresh = st.slider("Similarity threshold (min)", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    st.markdown("**Note:** If very few candidate items are covered by initial neighbors, the algorithm will *expand* neighbors automatically up to a safe limit to improve coverage.")

    st.subheader("Generate Recommendations")
    user_list = list(data_dict['user_to_idx'].keys())
    target_user_id = st.selectbox("Select a User ID to generate recommendations for:", options=user_list)

    if st.button("Get Recommendations"):
        with st.spinner("Generating recommendations..."):
            user_idx = data_dict['user_to_idx'][str(target_user_id)]
            recs = get_recommendations_user_based(
                user_idx,
                data_dict,
                nn_model,
                top_n=int(top_n),
                k_neighbors=int(k_neighbors),
                similarity_threshold=float(sim_thresh),
                max_k_expand=300
            )

        if recs:
            rec_df = pd.DataFrame(recs, columns=['Product Name', 'Predicted Rating'])
            st.write(f"### Top {top_n} recommendations for user {target_user_id}:")
            st.table(rec_df)
        else:
            st.info("No recommendations could be generated for this user (user may have no ratings, or neighbors didn't rate new items).")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
