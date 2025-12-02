import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Define Prediction and Recommendation Functions ---

def predict_rating_user_based(user_id, product_name, ratings_matrix, user_similarity_df):
    """Predicts a single rating using the Adjusted Weighted Sum formula."""
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
        
    # Clamp prediction to typical rating scale (e.g., 1 to 10 if that's the range)
    # Adjust range if your ratings are 1-5, for example
    return np.clip(prediction, 1, 10) 

def get_top_n_recommendations(user_id, ratings_matrix, user_similarity_df, n=5):
    """Generates a list of the top N recommended products for a given user."""
    user_ratings = ratings_matrix.loc[user_id]
    unrated_products = user_ratings[pd.isna(user_ratings)].index.tolist()
    
    if not unrated_products:
        return [("User has rated everything!", 0)]

    predictions = {}
    # Iterate over unrated items and make a prediction for each
    for product in unrated_products:
        predicted_rating = predict_rating_user_based(user_id, product, ratings_matrix, user_similarity_df)
        predictions[product] = predicted_rating
        
    # Sort predictions by rating in descending order
    top_recommendations = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    
    return top_recommendations[:n]

# --- 2. Streamlit UI Logic ---

st.title("Collaborative Filtering Recommendation System")

st.markdown("""
Upload your data file (CSV or Excel) with exactly three columns: 
**User ID, Product Name, and Rating.** 
The app will build a User-Based Collaborative Filter and suggest the top 5 products!
""")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Validate and rename columns
        if len(df.columns) != 3:
             st.error("Please ensure your file has exactly 3 columns.")
        else:
            df.columns = ['user_id', 'product_name', 'rating']
            
            st.success("File successfully loaded and columns internally renamed.")
            st.dataframe(df.head())

            # --- Data Processing ---
            st.subheader("Processing Data and Calculating Similarities...")
            
            # Create User-Item Matrix
            ratings_matrix = df.pivot_table(index='user_id', columns='product_name', values='rating')
            
            # Calculate User Similarity
            user_similarity = cosine_similarity(ratings_matrix.fillna(0))
            user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
            
            st.success(f"Matrix shape: {ratings_matrix.shape}. Similarity calculated.")

            # --- Recommendation Interface ---
            st.subheader("Generate Recommendations")
            
            # Dropdown for selecting a User ID
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
