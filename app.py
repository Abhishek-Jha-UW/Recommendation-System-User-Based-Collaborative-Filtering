import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# base64 import is no longer needed with this approach

# --- 1. Define Prediction and Recommendation Functions (Same as before) ---

def predict_rating_user_based(user_id, product_name, ratings_matrix, user_similarity_df):
    user_avg_rating = ratings_matrix.loc[user_id].mean()
    game_ratings = ratings_matrix[product_name]
    rated_users_ids = game_ratings[game_ratings.notna()].index.tolist()
    if not rated_users_ids:
        return user_avg_rating 
    similarities = user_similarity_df.loc[user_id, rated_users_ids]
    neighbors_avg_ratings = ratings_matrix.loc[rated_users_ids].mean(axis=1)
    ratings_diff = game_ratings[rated_users_ids] - neighbors_avg_ratings
    numerator = (similarities * ratings_diff).sum()
    denominator = similarities.abs().sum()
    if denominator == 0:
        prediction = user_avg_rating
    else:
        prediction = user_avg_rating + (numerator / denominator)
    return np.clip(prediction, 1, 10) 

def get_top_n_recommendations(user_id, ratings_matrix, user_similarity_df, n=5):
    user_ratings = ratings_matrix.loc[user_id]
    unrated_products = user_ratings[pd.isna(user_ratings)].index.tolist()
    if not unrated_products:
        return [("User has rated everything!", 0)]
    predictions = {}
    for product in unrated_products:
        predicted_rating = predict_rating_user_based(user_id, product, ratings_matrix, user_similarity_df)
        predictions[product] = predicted_rating
    top_recommendations = sorted(predictions.items(), key=lambda item: item, reverse=True)
    return top_recommendations[:n]

# --- 2. Streamlit UI Logic ---

st.set_page_config(page_title="CF Recommender", layout="centered")

st.title("Collaborative Filtering Recommendation System")

st.markdown("""
Upload your data file (CSV or Excel) with exactly three columns: 
**User ID, Product Name, and Rating.** 
The app will build a User-Based Collaborative Filter and suggest the top 5 products!
""")

st.markdown("---")
st.subheader("Try it out with a sample file:")

# URL points to the raw CSV file you just uploaded
github_csv_url = "raw.githubusercontent.com"

# Updated markdown to open the link in a new tab (target="_blank") using HTML
st.markdown(
    f'Download the sample file here: <a href="{github_csv_url}" target="_blank">games_sample.csv</a>', 
    unsafe_allow_html=True
)

st.markdown("---")


uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        if len(df.columns) != 3:
             st.error("Please ensure your file has exactly 3 columns.")
        else:
            df.columns = ['user_id', 'product_name', 'rating']
            
            st.success("File successfully loaded and columns internally renamed.")

            st.subheader("Processing Data and Calculating Similarities...")
            
            # --- Data Processing ---
            ratings_matrix = df.pivot_table(index='user_id', columns='product_name', values='rating')
            user_similarity = cosine_similarity(ratings_matrix.fillna(0))
            user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
            
            st.success(f"Matrix shape: {ratings_matrix.shape}. Similarity calculated.")

            st.subheader("Generate Recommendations")
            
            target_user = st.selectbox(
                "Select a User ID to generate recommendations for:",
                options=ratings_matrix.index.tolist()
            )
            
            if st.button("Get Top 5 Recommendations"):
                with st.spinner(f"Generating recommendations for User {target_user}..."):
                    recommendations = get_top_n_recommendations(target_user, ratings_matrix, user_similarity_df, n=5)
                    
                    st.write(f"### Top 5 Recommended Products for User {target_user}:")
                    rec_df = pd.DataFrame(recommendations, columns=['Product Name', 'Predicted Rating'])
                    st.table(rec_df)
            
    except Exception as e:
        st.error(f"An error occurred during file processing or calculation: {e}")

# --- 4. Add the Footer ---
st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
