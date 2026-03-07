# model.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from mlxtend.frequent_patterns import apriori, association_rules


# =========================================================
# 1. USER-BASED COLLABORATIVE FILTERING
# =========================================================

def compute_user_similarity(ratings_df, user_col="user_id", item_col="item_id", rating_col="rating"):
    """
    Creates a user-item matrix and computes cosine similarity between users.
    """
    user_item_matrix = ratings_df.pivot_table(
        index=user_col, columns=item_col, values=rating_col
    ).fillna(0)

    sparse_matrix = csr_matrix(user_item_matrix.values)
    similarity = cosine_similarity(sparse_matrix)

    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_item_matrix, similarity_df


def recommend_user_based(user_id, user_item_matrix, similarity_df, k=10):
    """
    Recommends items using User-Based Collaborative Filtering.
    """
    if user_id not in similarity_df.index:
        return []

    # Get similar users
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:11]

    # Weighted ratings
    weighted_scores = user_item_matrix.T.dot(similar_users)

    # Remove items already rated
    already_rated = user_item_matrix.loc[user_id]
    weighted_scores = weighted_scores[already_rated == 0]

    return weighted_scores.sort_values(ascending=False).head(k).index.tolist()



# =========================================================
# 2. ITEM-BASED COLLABORATIVE FILTERING
# =========================================================

def compute_item_similarity(ratings_df, user_col="user_id", item_col="item_id", rating_col="rating"):
    """
    Computes item-item similarity using cosine similarity.
    """
    user_item_matrix = ratings_df.pivot_table(
        index=user_col, columns=item_col, values=rating_col
    ).fillna(0)

    sparse_matrix = csr_matrix(user_item_matrix.values)
    similarity = cosine_similarity(sparse_matrix.T)

    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    return user_item_matrix, similarity_df


def recommend_item_based(user_id, user_item_matrix, item_similarity_df, k=10):
    """
    Recommends items using Item-Based Collaborative Filtering.
    """
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    scores = pd.Series(dtype=float)

    for item in rated_items:
        similar_items = item_similarity_df[item].sort_values(ascending=False)[1:50]
        scores = scores.add(similar_items * user_ratings[item], fill_value=0)

    # Remove already rated items
    scores = scores.drop(rated_items, errors="ignore")

    return scores.sort_values(ascending=False).head(k).index.tolist()



# =========================================================
# 3. MARKET BASKET ANALYSIS (APRIORI)
# =========================================================

def prepare_transactions(df, user_col="user_id", item_col="item_id"):
    """
    Converts transaction data into a one-hot encoded basket format.
    """
    basket = df.groupby([user_col, item_col]).size().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket


def run_market_basket(df, user_col="user_id", item_col="item_id",
                      min_support=0.01, metric="lift", min_threshold=1):
    """
    Runs Apriori and generates association rules.
    """
    basket = prepare_transactions(df, user_col, item_col)

    frequent_items = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)

    return rules


def recommend_market_basket(item, rules_df, k=10):
    """
    Recommends items based on association rules.
    """
    filtered = rules_df[rules_df['antecedents'].apply(lambda x: item in list(x))]

    if filtered.empty:
        return []

    filtered["consequents_str"] = filtered["consequents"].apply(lambda x: list(x)[0])
    return filtered.sort_values("lift", ascending=False)["consequents_str"].head(k).tolist()



# =========================================================
# 4. MASTER FUNCTION (OPTIONAL)
# =========================================================

def get_recommendations(method, **kwargs):
    """
    Unified interface for all recommendation methods.
    """
    if method == "user_based":
        return recommend_user_based(**kwargs)

    if method == "item_based":
        return recommend_item_based(**kwargs)

    if method == "market_basket":
        return recommend_market_basket(**kwargs)

    return []
