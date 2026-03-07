import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def compute_similarity(df, target='user', user_col="user_id", item_col="item_id", rating_col="rating"):
    # Pivot and fill zeros efficiently
    pivot = df.pivot_table(index=user_col, columns=item_col, values=rating_col).fillna(0)
    sparse_mat = csr_matrix(pivot.values)
    
    if target == 'item':
        # Item-Item Similarity
        sim = cosine_similarity(sparse_mat.T)
        return pivot, pd.DataFrame(sim, index=pivot.columns, columns=pivot.columns)
    else:
        # User-User Similarity
        sim = cosine_similarity(sparse_mat)
        return pivot, pd.DataFrame(sim, index=pivot.index, columns=pivot.index)

def recommend_user_based(user_id, pivot, sim_df, k=5):
    if user_id not in sim_df.index: return []
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:11]
    weights = pivot.loc[similar_users.index]
    scores = weights.T.dot(similar_users)
    
    # Filter items already rated by this user
    already_rated = pivot.loc[user_id]
    recommendations = scores[already_rated == 0]
    return recommendations.sort_values(ascending=False).head(k).index.tolist()

def recommend_item_based(user_id, pivot, sim_df, k=5):
    if user_id not in pivot.index: return []
    user_ratings = pivot.loc[user_id]
    active_items = user_ratings[user_ratings > 0].index
    
    # Sum similarities of all items user has rated
    scores = sim_df[active_items].sum(axis=1)
    recommendations = scores.drop(active_items, errors='ignore')
    return recommendations.sort_values(ascending=False).head(k).index.tolist()

def run_market_basket(df, user_col="user_id", item_col="item_id", min_support=0.01):
    # Convert to transaction matrix (1s and 0s)
    basket = df.groupby([user_col, item_col]).size().unstack().fillna(0)
    basket = (basket > 0).astype(int)
    
    n = float(len(basket))
    item_support = basket.sum() / n
    freq_items = item_support[item_support >= min_support]
    
    # High-speed Co-occurrence via Matrix Dot Product
    filtered_basket = basket[freq_items.index].values
    co_occurrence = np.dot(filtered_basket.T, filtered_basket)
    
    rules = []
    items = freq_items.index.tolist()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            support_ab = co_occurrence[i, j] / n
            if support_ab >= min_support:
                lift = support_ab / (freq_items[items[i]] * freq_items[items[j]])
                rules.append({"antecedent": items[i], "consequent": items[j], "lift": lift})
                rules.append({"antecedent": items[j], "consequent": items[i], "lift": lift})
    
    return pd.DataFrame(rules)
