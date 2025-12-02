import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys # Import sys for st.stop()

# --- 1. Define Prediction Function (Simplified & Robust) ---

def predict_rating_user_based(user_id, product_name, ratings_matrix, user_similarity_df):
    
    # 1. Get neighbors who actually rated this product
    game_ratings = ratings_matrix[product_name]
    rated_users_ids = game_ratings[game_ratings.notna()].index
    
    # 2. If no neighbors rated it, return None
    if rated_users_ids.empty or len(rated_users_ids) < 1:
        return None 
    
    # 3. Get similarities and filter to only the relevant neighbors
    similarities = user_similarity_df.loc[user_id, rated_users_ids]
    
    # 4. Handle cases where all relevant similarities are NaN or 0
    if similarities.sum() == 0 or similarities.isnull().all():
        return None

    # 5. Calculate weighted prediction using vectorized operations
    # Subtract neighbor's average from their specific rating for this item
    neighbors_avg = ratings_matrix.loc[rated_users_ids].mean(axis=1)
    ratings_diff = game_ratings[rated_users_ids] - neighbors_avg
    
    # Weight the difference by similarity score
    numerator = (similarities * ratings_diff).sum()
    denominator = similarities.abs().sum()
    
    # 6. Final prediction: Target user's average + weighted deviation
    user_avg_rating = ratings_matrix.loc[user_id].mean()
    prediction = user_avg_rating + (numerator / denominator)
        
    # Clamp rating between 1 and 10 (adjust this range if needed)
    return np.clip(prediction, 1, 10) 

# --- 2. Define Recommendation Function ---

def get_top_n_recommendations(user_id, ratings_matrix, user_similarity_df, n=5):
    user_ratings = ratings_matrix.loc[user_id]
    unrated_products = user_ratings[pd.isna(user_ratings)].index.tolist()
    
    if not unrated_products:
        return [] # Return empty list if everything is rated

    predictions = {}
    for product in unrated_products:
        predicted_rating = predict_rating_user_based(user_id, product, ratings_matrix, user_similarity_df)
        if predicted_rating is not None:
             predictions[product] = predicted_rating
        
    top_recommendations = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    
    return top_recommendations[:n]

# --- 3. Streamlit UI Logic ---

st.set_page_config(page_title="CF Recommender", layout="centered")
st.title("Collaborative Filtering Recommendation System")

st.markdown("""
Upload your data file (CSV or Excel) with exactly three columns: 
**User ID, Product Name, and Rating.** 
""")

st.markdown("---")
st.subheader("Try it out with a sample file:")
github_csv_url = "raw.githubusercontent.com"
st.markdown(
    f'Download the sample file here: <a href="{github_csv_url}" target="_blank" download="games_sample.csv">games_sample.csv</a>', 
    unsafe_allow_html=True
)
st.markdown("---")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Assumes standard comma-separated file format
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file) 
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        if len(df.columns) != 3:
             st.error("Please ensure your file has exactly 3 columns.")
             st.stop()
        else:
            df.columns = ['user_id', 'product_name', 'rating']
            st.success("File successfully loaded.")
            
            # --- Data Processing ---
            ratings_matrix = df.pivot_table(index='user_id', columns='product_name', values='rating')
            
            if ratings_matrix.shape[0] < 2:
                 st.error("Need at least 2 distinct users in the data to run collaborative filtering.")
                 st.stop()

            # Calculate Similarity
            user_similarity = cosine_similarity(ratings_matrix.fillna(0))
            user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
            st.success(f"Matrix shape: {ratings_matrix.shape}. Similarity calculated.")

            # --- Generate Recommendations UI ---
            st.subheader("Generate Recommendations")
            target_user = st.selectbox(
                "Select a User ID to generate recommendations for:",
                options=ratings_matrix.index.tolist()
            )
            
            if st.button("Get Top 5 Recommendations"):
                with st.spinner(f"Generating recommendations for User {target_user}..."):
                    recommendations = get_top_n_recommendations(target_user, ratings_matrix, user_similarity_df, n=5)
                    
                    if recommendations:
                        st.write(f"### Top 5 Recommended Products for User {target_user}:")
                        rec_df = pd.DataFrame(recommendations, columns=['Product Name', 'Predicted Rating'])
                        st.table(rec_df)
                    else:
                        st.info(f"No new recommendations could be generated for User {target_user}. (Might have rated everything or no overlapping neighbors found.)")
            
    except Exception as e:
        st.error(f"An unexpected error occurred during file processing or calculation: {e}")
        st.error("Please verify your file format is a standard CSV with commas.")

# --- 4. Add the Footer ---
st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
