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
    """Prepare sparse rating matrix and user/product mappings."""
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

    rating_csr = csr_matrix((ratings, (user_idx, prod_idx)), shape=(len(users), len(products)))

    user_counts = np.diff(rating_csr.indptr)
    user_sums = np.asarray(rating_csr.sum(axis=1)).ravel()
    user_means = np.zeros(len(users), dtype=float)
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
    """Build and cache a NearestNeighbors model."""
    nn = NearestNeighbors(metric='cosine', algorithm=algorithm, n_jobs=-1)
    nn.fit(_rating_csr)
    return nn

# ----------------------------
# Prediction logic
# ----------------------------

def predict_rating_user_based_sparse(user_idx, prod_idx, data_dict, nn_model, top_k=50, rating_min=1, rating_max=10):
    rating_csr = data_dict['rating_csr']
    user_means = data_dict['user_means']

    if data_dict['user_counts'][user_idx] == 0:
        return None

    users_who_rated = rating_csr[:, prod_idx].nonzero()[0]
    if users_who_rated.size == 0:
        return None

    n_users = rating_csr.shape[0]
    k_query = min(n_users, max(10, top_k + 5))
    
    distances, neighbor_indices = nn_model.kneighbors(rating_csr[user_idx], n_neighbors=k_query, return_distance=True)
    distances = np.ravel(distances)
    neighbor_indices = np.ravel(neighbor_indices)
    neigh_sims = 1.0 - distances

    mask_rated = np.isin(neighbor_indices, users_who_rated)
    filtered_neighbors = neighbor_indices[mask_rated]
    filtered_sims = neigh_sims[mask_rated]

    if filtered_neighbors.size == 0:
        return None

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

    prediction = user_means[user_idx] + (numerator / denominator)
    return float(np.clip(prediction, rating_min, rating_max))

def get_top_n_recommendations_sparse(user_idx, data_dict, nn_model, n=5, top_k_neighbors=50):
    rating_csr = data_dict['rating_csr']
    user_row = rating_csr[user_idx]
    unrated_mask = (user_row.toarray().ravel() == 0)
    unrated_prod_idx = np.where(unrated_mask)[0].tolist()

    preds = {}
    for p_idx in unrated_prod_idx:
        pred = predict_rating_user_based_sparse(user_idx, p_idx, data_dict, nn_model, top_k=top_k_neighbors)
        if pred is not None:
            preds[p_idx] = pred

    top_items = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]
    idx_to_prod = data_dict['idx_to_prod']
    return [(idx_to_prod[idx], rating) for idx, rating in top_items]

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="CF Recommender", layout="centered")
st.title("User-Based Collaborative Filtering Recommender")

st.markdown("""
The app uses sparse matrices + KNN for fast and accurate recommendations.  
You can use the **default sample dataset** or upload your **own CSV/Excel** file.
""")

st.markdown("---")

# ----------------------------
# File selection: default GitHub sample or upload
# ----------------------------

uploaded_file = st.file_uploader("Upload your CSV/Excel file (Optional)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.success("Uploaded file loaded.")
else:
    github_csv_url = "https://raw.githubusercontent.com/Abhishek-Jha-UW/Recommendation-System-User-Based-Collaborative-Filtering/main/games_sample.csv"
    df = pd.read_csv(github_csv_url)
    st.success("Sample dataset loaded from GitHub.")

df.columns = ['user_id', 'product_name', 'rating']

# ----------------------------
# Prepare matrices and KNN model
# ----------------------------
with st.spinner("Preparing sparse matrices and KNN model..."):
    data_dict = prepare_matrices(df)
    rating_csr = data_dict['rating_csr']
    n_users, n_products = rating_csr.shape

    if n_users < 2:
        st.error("Need at least 2 distinct users to run CF.")
        st.stop()

    nn_model = build_nn_model(rating_csr)

st.write(f"Users: {n_users} — Products: {n_products} — Ratings (non-zero): {rating_csr.nnz}")

# ----------------------------
# Generate Recommendations
# ----------------------------
st.subheader("Generate Recommendations")
user_list = list(data_dict['user_to_idx'].keys())
target_user_id = st.selectbox("Select a User ID to generate recommendations for:", options=user_list)

if st.button("Get Top 5 Recommendations"):
    with st.spinner(f"Generating recommendations for User {target_user_id}..."):
        user_idx = data_dict['user_to_idx'][target_user_id]
        recommendations = get_top_n_recommendations_sparse(user_idx, data_dict, nn_model, n=5)
        if recommendations:
            rec_df = pd.DataFrame(recommendations, columns=['Product Name', 'Predicted Rating'])
            st.write(f"### Top 5 Recommended Products for User {target_user_id}:")
            st.table(rec_df)
        else:
            st.info("No recommendations could be generated.")

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
