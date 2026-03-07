import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Compute Similarity Matrix
# -----------------------------
def compute_similarity(df, target="user",
                       user_col="user_id",
                       item_col="item_id",
                       rating_col="rating"):

    pivot = df.pivot(index=user_col, columns=item_col, values=rating_col).fillna(0)

    if target == "item":
        sim_matrix = cosine_similarity(pivot.T)
        sim_df = pd.DataFrame(sim_matrix,
                              index=pivot.columns,
                              columns=pivot.columns)
    else:
        sim_matrix = cosine_similarity(pivot)
        sim_df = pd.DataFrame(sim_matrix,
                              index=pivot.index,
                              columns=pivot.index)

    return pivot, sim_df


# -----------------------------
# User-Based Recommendation
# -----------------------------
def recommend_user_based(user_id, pivot, sim_df, k=5):

    if user_id not in pivot.index:
        return []

    similar_users = sim_df[user_id].sort_values(ascending=False)[1:11]

    weighted_scores = np.dot(similar_users.values,
                             pivot.loc[similar_users.index])

    scores = pd.Series(weighted_scores, index=pivot.columns)

    rated_items = pivot.loc[user_id]
    recommendations = scores[rated_items == 0]

    return recommendations.sort_values(ascending=False).head(k).index.tolist()


# -----------------------------
# Item-Based Recommendation
# -----------------------------
def recommend_item_based(user_id, pivot, sim_df, k=5):

    if user_id not in pivot.index:
        return []

    user_ratings = pivot.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    if len(rated_items) == 0:
        return []

    scores = sim_df[rated_items].sum(axis=1)

    recommendations = scores.drop(rated_items, errors="ignore")

    return recommendations.sort_values(ascending=False).head(k).index.tolist()


# -----------------------------
# Market Basket Analysis
# -----------------------------
def run_market_basket(df,
                      user_col="user_id",
                      item_col="item_id",
                      min_support=0.01):

    basket = df.groupby([user_col, item_col]).size().unstack(fill_value=0)

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    item_support = basket.sum() / len(basket)

    frequent_items = item_support[item_support >= min_support].index

    basket = basket[frequent_items]

    co_matrix = basket.T.dot(basket)

    rules = []

    for item_a in frequent_items:
        for item_b in frequent_items:

            if item_a == item_b:
                continue

            support_ab = co_matrix.loc[item_a, item_b] / len(basket)

            if support_ab >= min_support:

                lift = support_ab / (
                    item_support[item_a] * item_support[item_b]
                )

                rules.append({
                    "antecedent": item_a,
                    "consequent": item_b,
                    "lift": lift
                })

    rules_df = pd.DataFrame(rules)

    if not rules_df.empty:
        rules_df = rules_df.sort_values("lift", ascending=False)

    return rules_df
