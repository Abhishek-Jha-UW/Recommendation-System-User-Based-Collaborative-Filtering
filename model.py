import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(pivot, target="user"):
    """Computes cosine similarity matrix for users or items."""
    # Fill NA with 0 only for the similarity calculation
    filled_pivot = pivot.fillna(0)
    
    if target == "item":
        sim_matrix = cosine_similarity(filled_pivot.T)
        idx = pivot.columns
    else:
        sim_matrix = cosine_similarity(filled_pivot)
        idx = pivot.index
    
    return pd.DataFrame(sim_matrix, index=idx, columns=idx)

def predict_rating_advanced(user_id, item_id, pivot, user_sim):
    """Predicts a rating using Mean-Centered Collaborative Filtering."""
    if item_id not in pivot.columns:
        return None
    
    # Get users who actually rated this specific item
    item_ratings = pivot[item_id].dropna()
    if item_ratings.empty:
        return None
    
    # Find similarities between target user and those who rated the item
    sims = user_sim.loc[user_id, item_ratings.index]
    if sims.sum() == 0:
        return None
    
    # Math: User Mean + [Sum(Sim * (Rating - Neighbor_Mean)) / Sum(Sim)]
    user_mean = pivot.loc[user_id].mean()
    neighbor_means = pivot.loc[item_ratings.index].mean(axis=1)
    
    weighted_diff = np.dot(sims, (item_ratings - neighbor_means))
    prediction = user_mean + (weighted_diff / sims.sum())
    
    # Assuming a 1-10 scale (adjust to 1, 5 if your data is 1-5)
    return np.clip(prediction, 1, 10)

def get_recommendations(user_id, pivot, user_sim, n=5):
    """Returns top N recommended items for a user."""
    if user_id not in pivot.index:
        return []
        
    user_ratings = pivot.loc[user_id]
    # We only want to predict for items the user HAS NOT rated yet
    unrated_items = user_ratings[user_ratings.isna()].index
    
    predictions = []
    for item in unrated_items:
        score = predict_rating_advanced(user_id, item, pivot, user_sim)
        if score:
            predictions.append((item, score))
            
    # Sort by predicted score descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]
