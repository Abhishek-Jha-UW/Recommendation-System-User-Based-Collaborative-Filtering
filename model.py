import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# =========================================================
# USER-BASED COLLABORATIVE FILTERING
# =========================================================

def compute_user_similarity(df, user_col="user_id", item_col="item_id", rating_col="rating"):
    user_item = df.pivot_table(index=user_col, columns=item_col, values=rating_col).fillna(0)
    sparse = csr_matrix(user_item.values)
    sim = cosine_similarity(sparse)
    sim_df = pd.DataFrame(sim, index=user_item.index, columns=user_item.index)
    return user_item, sim_df


def recommend_user_based(user_id, user_item, sim_df, k=5):
    if user_id not in sim_df.index:
        return []

    similar_users = sim_df[user_id].sort_values(ascending=False)[1:11]
    weighted_scores = user_item.T.dot(similar_users)

    already_rated = user_item.loc[user_id]
    weighted_scores = weighted_scores[already_rated == 0]

    return weighted_scores.sort_values(ascending=False).head(k).index.tolist()


# =========================================================
# ITEM-BASED COLLABORATIVE FILTERING
# =========================================================

def compute_item_similarity(df, user_col="user_id", item_col="item_id", rating_col="rating"):
    user_item = df.pivot_table(index=user_col, columns=item_col, values=rating_col).fillna(0)
    sparse = csr_matrix(user_item.values)
    sim = cosine_similarity(sparse.T)
    sim_df = pd.DataFrame(sim, index=user_item.columns, columns=user_item.columns)
    return user_item, sim_df


def recommend_item_based(user_id, user_item, sim_df, k=5):
    if user_id not in user_item.index:
        return []

    user_ratings = user_item.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index

    scores = pd.Series(dtype=float)

    for item in rated_items:
        similar_items = sim_df[item].sort_values(ascending=False)[1:50]
        scores = scores.add(similar_items * user_ratings[item], fill_value=0)

    scores = scores.drop(rated_items, errors="ignore")
    return scores.sort_values(ascending=False).head(k).index.tolist()


# =========================================================
# MARKET BASKET ANALYSIS (NO MLXTEND)
# =========================================================

def prepare_transactions(df, user_col="user_id", item_col="item_id"):
    basket = df.groupby([user_col, item_col]).size().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket


def fpgrowth(basket, min_support=0.01):
    n = len(basket)

    item_support = basket.sum() / n
    freq_items = item_support[item_support >= min_support]

    pairs = {}
    items = freq_items.index.tolist()

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            support = (basket[items[i]] & basket[items[j]]).sum() / n
            if support >= min_support:
                pairs[(items[i], items[j])] = support

    return freq_items.to_dict(), pairs


def generate_rules(freq_items, freq_pairs, min_lift=1.0):
    rules = []

    for (a, b), support_ab in freq_pairs.items():
        lift_ab = support_ab / (freq_items[a] * freq_items[b])

        if lift_ab >= min_lift:
            rules.append({"antecedent": a, "consequent": b, "lift": lift_ab})
            rules.append({"antecedent": b, "consequent": a, "lift": lift_ab})

    return pd.DataFrame(rules)


def run_market_basket(df, user_col="user_id", item_col="item_id", min_support=0.01, min_lift=1.0):
    basket = prepare_transactions(df, user_col, item_col)
    freq_items, freq_pairs = fpgrowth(basket, min_support)
    rules = generate_rules(freq_items, freq_pairs, min_lift)
    return rules


def recommend_market_basket(item, rules_df, k=5):
    filtered = rules_df[rules_df["antecedent"] == item]
    if filtered.empty:
        return []
    return filtered.sort_values("lift", ascending=False).head(k)["consequent"].tolist()
