# cf_recommender_final_v2.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests
from io import StringIO

# ----------------------------
# Config
# ----------------------------
GITHUB_RAW_SAMPLE = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"

# ----------------------------
# Cached helpers
# ----------------------------
@st.cache_data
def load_sample_csv(url):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))

@st.cache_data
def prepare_matrices(df):
    df = df.copy()
    df['user_id'] = df['user_id'].astype(str)
    df['product_name'] = df['product_name'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating', 'user_id', 'product_name']).reset_index(drop=True)

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
        'user_means': user_means
    }

@st.cache_resource
def build_nn_model(_rating_csr, algorithm='brute'):
    nn = NearestNeighbors(metric='cosine', algorithm=algorithm, n_jobs=-1)
    nn.fit(_rating_csr)
    return nn

# ----------------------------
# User-based CF prediction
# ----------------------------
def predict_user_based(user_idx, prod_idx, data_dict, nn_model, top_k=100):
    rating_csr = data_dict['rating_csr']
    user_means = data_dict['user_means']
    n_users = rating_csr.shape[0]

    # Users who rated this product
    users_who_rated = rating_csr[:, prod_idx].nonzero()[0]
    if len(users_who_rated) == 0:
        return None

    # Query neighbors
    k_query = min(n_users, top_k + 10)
    distances, neighbors = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=k_query, return_distance=True)
    distances = np.ravel(distances)
    neighbors = np.ravel(neighbors)

    # Remove self
    mask_self = neighbors != user_idx
    neighbors = neighbors[mask_self]
    distances = distances[mask_self]

    if len(neighbors) == 0:
        return None

    sims = 1.0 - distances

    # Only consider neighbors who rated the product
    mask_rated = np.isin(neighbors, users_who_rated)
    neighbors = neighbors[mask_rated]
    sims = sims[mask_rated]

    if len(neighbors) == 0:
        return None

    neighbor_ratings = rating_csr[neighbors, prod_idx].toarray().ravel()
    neighbor_means = user_means[neighbors]
    ratings_diff = neighbor_ratings - neighbor_means

    numerator = np.dot(sims, ratings_diff)
    denominator = np.sum(np.abs(sims))
    if denominator == 0:
        return None

    prediction = user_means[user_idx] + (numerator / denominator)
    # Clip prediction to observed min/max ratings
    all_ratings = rating_csr.data
    rmin, rmax = float(all_ratings.min()), float(all_ratings.max())
    return float(np.clip(prediction, rmin, rmax))

def get_top_n_recommendations(user_idx, data_dict, nn_model, n=5, top_k=100):
    rating_csr = data_dict['rating_csr']
    user_row = rating_csr[user_idx].toarray().ravel()
    unrated_idx = np.where(user_row == 0)[0]
    preds = {}
    for p_idx in unrated_idx:
        pred = predict_user_based(user_idx, p_idx, data_dict, nn_model, top_k)
        if pred is not None:
            preds[p_idx] = pred
    top_idx = sorted(preds, key=preds.get, reverse=True)[:n]
    idx_to_prod = data_dict['idx_to_prod']
    return [(idx_to_prod[i], round(preds[i], 4)) for i in top_idx]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="User-Based CF Recommender", layout="centered")
st.title("Collaborative Filtering — User-Based (Optimized)")

# Load GitHub sample
with st.spinner("Loading sample CSV from GitHub..."):
    try:
        df = load_sample_csv(GITHUB_RAW_SAMPLE)
        st.success("Sample CSV loaded.")
    except Exception as e:
        st.error(f"Cannot load sample CSV: {e}")
        st.stop()

# Optional file upload
uploaded_file = st.file_uploader("Or upload your own CSV/XLSX file", type=["csv","xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.success("Uploaded file loaded.")

# Validate
if len(df.columns) != 3:
    st.error("CSV must have 3 columns: User ID, Product Name, Rating")
    st.stop()
df.columns = ['user_id', 'product_name', 'rating']

# Prepare matrices
with st.spinner("Preparing matrices and KNN model..."):
    data_dict = prepare_matrices(df)
    nn_model = build_nn_model(data_dict['rating_csr'])
    n_users, n_products = data_dict['rating_csr'].shape
st.write(f"Users: {n_users} — Products: {n_products} — Ratings: {data_dict['rating_csr'].nnz}")

# Parameters
st.subheader("Recommendation Parameters")
col1, col2 = st.columns(2)
with col1:
    top_n = st.number_input("Top-N recommendations", min_value=1, max_value=50, value=5)
with col2:
    top_k = st.number_input("Number of neighbors (K)", min_value=5, max_value=n_users, value=100)

# Generate recommendations
st.subheader("Generate Recommendations")
target_user = st.selectbox("Select User ID:", list(data_dict['user_to_idx'].keys()))

if st.button("Get Top Recommendations"):
    user_idx = data_dict['user_to_idx'][str(target_user)]
    recs = get_top_n_recommendations(user_idx, data_dict, nn_model, n=int(top_n), top_k=int(top_k))
    if recs:
        rec_df = pd.DataFrame(recs, columns=['Product Name','Predicted Rating'])
        st.write(f"### Top {top_n} Recommendations for User {target_user}")
        st.table(rec_df)
    else:
        st.info("No recommendations could be generated for this user.")

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
