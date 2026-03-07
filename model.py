import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# =========================================================
# COLLABORATIVE FILTERING (Optimized)
# =========================================================

def compute_similarity(df, target='user', user_col="user_id", item_col="item_id", rating_col="rating"):
    """
    Combined similarity function to reduce redundancy.
    target='user' for User-User, target='item' for Item-Item.
    """
    # Use pivot instead of pivot_table if there are no duplicate user-item pairs (faster)
    user_item = df.pivot_table(index=user_col, columns=item_col, values=rating_col).fillna(0)
    
    # Convert to sparse matrix for memory efficiency
    sparse_mat = csr_matrix(user_item.values)
    
    # If item-based, we transpose the matrix
    if target == 'item':
        sim = cosine_similarity(sparse_mat.T)
        return user_item, pd.DataFrame(sim, index=user_item.columns, columns=user_item.columns)
    else:
        sim = cosine_similarity(sparse_mat)
        return user_item, pd.DataFrame(sim, index=user_item.index, columns=user_item.index)

def recommend_user_based(user_id, user_item, sim_df, k=5):
    if user_id not in sim_df.index:
        return []

    # Get top 10 similar users
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:11]
    
    # Dot product of similar users' ratings and their similarity scores
    user_weights = user_item.loc[similar_users.index]
    weighted_scores = user_weights.T.dot(similar_users)

    # Filter out items the user has already seen/rated
    already_rated = user_item.loc[user_id]
    recommendations = weighted_scores[already_rated == 0]

    return recommendations.sort_values(ascending=False).head(k).index.tolist()

# =========================================================
# MARKET BASKET ANALYSIS (Corrected)
# =========================================================

def prepare_transactions(df, user_col="user_id", item_col="item_id"):
    basket = df.groupby([user_col, item_col]).size().unstack().fillna(0)
    # applymap is deprecated in newer pandas; use map for element-wise
    basket = basket.map(lambda x: 1 if x > 0 else 0)
    return basket

def simple_apriori(basket, min_support=0.01):
    """
    Renamed from fpgrowth as this is technically a manual Apriori-style 
    frequent pair counter. Added efficiency for large datasets.
    """
    n = float(len(basket))
    item_support = basket.sum() / n
    freq_items = item_support[item_support >= min_support]
    
    pairs = {}
    items = freq_items.index.tolist()

    # Optimization: Use matrix multiplication to find concurrent transactions
    # This is MUCH faster than nested loops for large datasets
    basket_filtered = basket[items].values
    co_occurrence = np.dot(basket_filtered.T, basket_filtered)
    
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            support_ab = co_occurrence[i, j] / n
            if support_ab >= min_support:
                pairs[(items[i], items[j])] = support_ab

    return freq_items.to_dict(), pairs

# ... (generate_rules remains largely the same, but ensure it handles empty freq_pairs)
