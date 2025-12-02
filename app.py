import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# base64 import is no longer needed with this approach

# --- 1. Define Prediction and Recommendation Functions (Enhanced Error Handling) ---

def predict_rating_user_based(user_id, product_name, ratings_matrix, user_similarity_df):
    """Predicts a single rating using the Adjusted Weighted Sum formula, with added checks."""
    user_avg_rating = ratings_matrix.loc[user_id].mean()
    
    # Check if the game exists in the matrix columns
    if product_name not in ratings_matrix.columns:
        return user_avg_rating # Cannot predict, return average as a fallback

    game_ratings = ratings_matrix[product_name]
    rated_users_ids = game_ratings[game_ratings.notna()].index.tolist()
    
    # CRITICAL CHECK: If no other users rated this game, we can't collaborate
    if not rated_users_ids or len(rated_users_ids) == 0:
        return user_avg_rating # Fallback to user's average if no overlap

    # Get the similarity scores between the target user and the neighbors
    similarities = user_similarity_df.loc[user_id, rated_users_ids]
    
    # Filter out neighbors with 0 or NaN similarity
    similarities = similarities[similarities.notna() & (similarities > 0)]

    if similarities.empty:
        return user_avg_rating

    # Calculate the average ratings for the neighbors
    neighbors_avg_ratings = ratings_matrix.loc[similarities.index].mean(axis=1)
    
    # Calculate the difference: Neighbor's rating for the game minus their average rating
    ratings_diff = game_ratings[similarities.index] - neighbors_avg_ratings
    
    # Calculate the numerator (Sum of (Similarity * Difference))
    numerator = (similarities * ratings_diff).sum()
    
    # Calculate the denominator (Sum of absolute similarities)
    denominator = similarities.abs().sum()
    
    # Handle division by zero edge case
    if denominator == 0:
        prediction = user_avg_rating
    else:
        prediction = user_avg_rating + (numerator / denominator)
        
    return np.clip(prediction, 1, 10) 


def get_top_n_recommendations(user_id, ratings_matrix, user_similarity_df, n=5):
    """Generates a list of the top N recommended products for a given user."""
    
    if user_id not in ratings_matrix.index:
        st.error(f"User ID {user_id} not found in the matrix.")
        return []

    user_ratings = ratings_matrix.loc[user_id]
    unrated_products = user_ratings[pd.isna(user_ratings)].index.tolist()
    
    if not unrated_products:
        return [("User has rated everything!", 0)]

    predictions = {}
    for product in unrated_products:
        # The enhanced predict function should now handle edge cases safely
        predicted_rating = predict_rating_user_based(user_id, product, ratings_matrix, user_similarity_df)
        if predicted_rating is not None:
             predictions[product] = predicted_rating
        
    top_recommendations = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    
    return top_recommendations[:n]

# --- 2. Streamlit UI Logic (Same as before) ---

st.set_page_config(page_title="CF Recommender", layout="centered")
st.title("Collaborative Filtering Recommendation System")
st.markdown("""
Upload your data file (CSV or Excel) with exactly three columns: 
**User ID, Product Name, and Rating.** 
The app will build a User-Based Collaborative Filter and suggest the top 5 products!
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
        if uploaded_file.name.endswith('.csv'):
            # Using default comma separator
            df = pd.read_csv(uploaded_file) 
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        if len(df.columns) != 3:
             st.error("Please ensure your file has exactly 3 columns.")
             st.stop()
        else:
            df.columns = ['user_id', 'product_name', 'rating']
            st.success("File successfully loaded and columns internally renamed.")
            st.subheader("Processing Data and Calculating Similarities...")
            
            ratings_matrix = df.pivot_table(index='user_id', columns='product_name', values='rating')
            if ratings_matrix.shape[0] < 2:
                 st.error("Need at least 2 users to calculate similarity. Please upload a larger dataset.")
                 st.stop()

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
                    
                    if recommendations and recommendations[0][0] != "User has rated everything!":
                        st.write(f"### Top 5 Recommended Products for User {target_user}:")
                        rec_df = pd.DataFrame(recommendations, columns=['Product Name', 'Predicted Rating'])
                        st.table(rec_df)
                    else:
                        st.info(f"User {target_user} has already rated all available products in the uploaded dataset.")
            
    except Exception as e:
        st.error(f"An error occurred during file processing or calculation: {e}")
        st.error(f"Please check file format. Original error: {e}")

# --- 4. Add the Footer ---
st.markdown("---")
st.markdown("Made with 💗 by Abhishek Jha", unsafe_allow_html=True)
