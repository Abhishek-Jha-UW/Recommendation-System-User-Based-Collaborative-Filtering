# cf_recommender_final.py
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
        'user_sums': user_sums,
        'user_means': user_means
    }

@st.cache_resource
def build_nn_model(_rating_csr, algorithm='brute'):
    nn = NearestNeighbors(metric='cosine', algorithm=algorithm, n_jobs=-1)
    nn.fit(_rating_csr)
    return nn

# ----------------------------
# Robust adaptive CF function
# ----------------------------
def get_recommendations_user_based(user_idx,
                                   data_dict,
                                   nn_model,
                                   top_n=5,
                                   k_neighbors=50,
                                   similarity_threshold=0.0,
                                   max_k_expand=300):
    rating_csr = data_dict['rating_csr']
    user_means = data_dict['user_means']
    n_users, n_products = rating_csr.shape

    def kneigh(kq):
        kq = min(max(2, int(kq)), n_users)
        distances, neighbors = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=kq, return_distance=True)
        return np.ravel(distances), np.ravel(neighbors)

    k_query = min(k_neighbors + 1, n_users)
    distances, neighbors = kneigh(k_query)
    mask_self = neighbors != user_idx
    neighbors = neighbors[mask_self]
    distances = distances[mask_self]

    if neighbors.size == 0:
        return []

    sims = 1.0 - distances
    good_mask = sims >= similarity_threshold
    neighbors = neighbors[good_mask]
    sims = sims[good_mask]

    if neighbors.size == 0:
        distances, neighbors = kneigh(k_query)
        mask_self = neighbors != user_idx
        neighbors = neighbors[mask_self]
        distances = distances[mask_self]
        sims = 1.0 - distances

    if neighbors.size == 0:
        return []

    if neighbors.size > k_neighbors:
        sorted_idx = np.argsort(sims)[-k_neighbors:]
        neighbors = neighbors[sorted_idx]
        sims = sims[sorted_idx]

    neighbor_dense = rating_csr[neighbors].toarray()
    neighbor_means = data_dict['user_means'][neighbors]
    deviations = neighbor_dense - neighbor_means.reshape(-1, 1)
    sims = np.maximum(sims, 0.0)
    weights = sims.reshape(-1, 1)
    weighted_dev = (deviations * weights).sum(axis=0)
    rated_mask_by_neighbor = neighbor_dense != 0
    denom = (rated_mask_by_neighbor * np.abs(weights)).sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_dev = np.zeros_like(weighted_dev, dtype=float)
        nonzero = denom > 0
        pred_dev[nonzero] = weighted_dev[nonzero] / denom[nonzero]

    target_user_mean = data_dict['user_means'][user_idx]
    preds = target_user_mean + pred_dev

    all_obs = data_dict['rating_csr'].data
    if all_obs.size > 0:
        rmin, rmax = float(all_obs.min()), float(all_obs.max())
    else:
        rmin, rmax = 1.0, 10.0
    preds = np.clip(preds, rmin, rmax)

    user_row = rating_csr[user_idx].toarray().ravel()
    preds[user_row != 0] = -np.inf

    top_idx = np.argsort(preds)[-top_n:][::-1]
    top_idx = [int(i) for i in top_idx if preds[i] != -np.inf]
    idx_to_prod = data_dict['idx_to_prod']
    recs = [(idx_to_prod[i], float(round(preds[i], 4))) for i in top_idx]
    return recs

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CF Recommender (GitHub Sample)", layout="centered")
st.title("Collaborative Filtering — User-Based (Robust)")

st.markdown("This app uses your GitHub `games_sample.csv` by default, or you can upload your own CSV/XLSX file.")

# Load GitHub sample by default
with st.spinner("Loading GitHub sample CSV..."):
    try:
        df = load_sample_csv(GITHUB_RAW_SAMPLE)
        st.success("Sample CSV loaded from GitHub.")
    except Exception as e:
        st.error(f"Could not load sample CSV: {e}")
        st.stop()

# Optional file upload
uploaded_file = st.file_uploader("Or upload your own CSV/XLSX file:", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("Uploaded file loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()

# Validate
if len(df.columns) != 3:
    st.error("CSV must have exactly 3 columns: User ID, Product Name, Rating")
    st.stop()
df.columns = ['user_id', 'product_name', 'rating']

with st.spinner("Preparing matrices and building KNN..."):
    data_dict = prepare_matrices(df)
    rating_csr = data_dict['rating_csr']
    nn_model = build_nn_model(rating_csr)
    n_users, n_products = rating_csr.shape
st.write(f"Users: {n_users} — Products: {n_products} — Ratings: {rating_csr.nnz}")

# Recommendation parameters
st.subheader("Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    top_n = st.number_input("Top-N", min_value=1, max_value=50, value=5)
with col2:
    k_neighbors = st.number_input("Initial K neighbors", min_value=5, max_value=500, value=50)
with col3:
    sim_thresh = st.slider("Min similarity threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

st.subheader("Generate Recommendations")
user_list = list(data_dict['user_to_idx'].keys())
target_user_id = st.selectbox("Select User ID:", user_list)

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
        st.write(f"### Top {top_n} recommendations for {target_user_id}")
        st.table(rec_df)
    else:
        st.info("No recommendations could be generated for this user.")

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
