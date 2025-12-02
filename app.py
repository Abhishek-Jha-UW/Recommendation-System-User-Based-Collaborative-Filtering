# cf_recommender_final_dense.py
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

st.set_page_config(page_title="User-Based CF Recommender", layout="centered")
st.title("Collaborative Filtering — User-Based (Dense)")

# ----------------------------
# Load sample CSV from GitHub
# ----------------------------
df = None
try:
    resp = requests.get(GITHUB_RAW_SAMPLE, timeout=10)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    st.success("Loaded sample CSV from GitHub")
except Exception as e:
    st.warning(f"Cannot load sample CSV from GitHub: {e}")

# ----------------------------
# Optional file upload
# ----------------------------
uploaded_file = st.file_uploader("Or upload your own CSV/XLSX file", type=["csv","xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.success("Uploaded file loaded")
    except Exception as e:
        st.error(f"Failed to load uploaded file: {e}")

# ----------------------------
# Validate and prepare data
# ----------------------------
if df is None:
    st.stop()

if len(df.columns) != 3:
    st.error("CSV must have exactly 3 columns: User ID, Product Name, Rating")
    st.stop()

df.columns = ['user_id','product_name','rating']
df['user_id'] = df['user_id'].astype(str)
df['product_name'] = df['product_name'].astype(str)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['user_id','product_name','rating'])

ratings_matrix = df.pivot_table(index='user_id', columns='product_name', values='rating')
user_similarity_df = pd.DataFrame(
    cosine_similarity(ratings_matrix.fillna(0)),
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

st.write(f"Users: {ratings_matrix.shape[0]} — Products: {ratings_matrix.shape[1]}")

# ----------------------------
# Prediction functions
# ----------------------------
def predict_rating_user_based(user_id, product_name):
    if product_name not in ratings_matrix.columns or user_id not in ratings_matrix.index:
        return None
    rated_users = ratings_matrix[product_name].dropna().index
    if len(rated_users)==0:
        return None
    sims = user_similarity_df.loc[user_id, rated_users]
    if sims.sum()==0:
        return None
    neighbors_avg = ratings_matrix.loc[rated_users].mean(axis=1)
    ratings_diff = ratings_matrix.loc[rated_users, product_name] - neighbors_avg
    numerator = (sims * ratings_diff).sum()
    denominator = sims.abs().sum()
    user_avg = ratings_matrix.loc[user_id].mean()
    prediction = user_avg + numerator/denominator
    # Clip to min/max observed ratings
    rmin, rmax = ratings_matrix.min().min(), ratings_matrix.max().max()
    return float(np.clip(prediction, rmin, rmax))

def get_top_n_recommendations(user_id, n=5):
    unrated = ratings_matrix.loc[user_id][ratings_matrix.loc[user_id].isna()].index
    preds = {}
    for p in unrated:
        pred = predict_rating_user_based(user_id, p)
        if pred is not None:
            preds[p] = pred
    top = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:n]
    return [(product, round(score, 4)) for product, score in top]

# ----------------------------
# UI for parameters and user selection
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    top_n = st.number_input("Top-N recommendations", min_value=1, max_value=20, value=5)
with col2:
    default_user = ratings_matrix.index[0] if len(ratings_matrix.index)>0 else None
    target_user = st.selectbox("Select User ID", ratings_matrix.index.tolist(), index=0)

# ----------------------------
# Generate recommendations
# ----------------------------
st.subheader("Top Recommendations")
if target_user:
    recs = get_top_n_recommendations(target_user, n=top_n)
    if recs:
        rec_df = pd.DataFrame(recs, columns=['Product Name','Predicted Rating'])
        st.table(rec_df)
    else:
        st.info("No recommendations could be generated for this user.")

st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
