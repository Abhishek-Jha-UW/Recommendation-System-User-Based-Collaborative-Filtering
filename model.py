import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(pivot, target="user"):
    # Fill NA with 0 for cosine similarity calculation
    filled_pivot = pivot.fillna(0)
    if target == "item":
        sim_matrix = cosine_similarity(filled_pivot.T)
    else:
        sim_matrix = cosine_similarity(filled_pivot)
    
    return pd.DataFrame(sim_matrix, index=sim_matrix_index(pivot, target), columns=sim_matrix_index(pivot, target))

def sim_matrix_index(pivot, target):
    return pivot.columns if target == "item" else pivot.index

def predict_rating_advanced(user_id, item_id, pivot, user_sim):
    if item_id not in pivot.columns:
        return None
    
    # Get users who rated this item
    item_ratings = pivot[item_id].dropna()
    if item_ratings.empty:
        return None
    
    # Get similarities between target user and those who rated the item
    sims = user_sim.loc[user_id, item_ratings.index]
    if sims.sum() == 0:
        return None
    
    # Advanced Mean-Centering Logic
    user_mean = pivot.loc[user_id].mean()
    neighbor_means = pivot.loc[item_ratings.index].mean(axis=1)
    
    # (Rating - Neighbor Mean) weighted by Similarity
    weighted_diff = np.dot(sims, (item_ratings - neighbor_means))
    prediction = user_mean + (weighted_diff / sims.sum())
    
    return np.clip(prediction, 1, 10)

def get_recommendations(user_id, pivot, user_sim, n=5):
    user_ratings = pivot.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index
    
    predictions = []
    for item in unrated_items:
        score = predict_rating_advanced(user_id, item, pivot, user_sim)
        if score:
            predictions.append((item, score))
            
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]
