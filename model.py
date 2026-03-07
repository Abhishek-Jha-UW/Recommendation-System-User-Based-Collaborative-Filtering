# =========================================================
# MARKET BASKET ANALYSIS (FP-GROWTH, NO MLXTEND)
# =========================================================

def prepare_transactions(df, user_col="user_id", item_col="item_id"):
    """
    Converts transaction data into a one-hot encoded basket format.
    """
    basket = df.groupby([user_col, item_col]).size().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    return basket


def fpgrowth(basket, min_support=0.01):
    """
    Simple FP-Growth-like frequent itemset generator using pandas.
    Supports 1-item and 2-item sets (sufficient for recommendations).
    """
    num_transactions = len(basket)

    # 1-item support
    item_support = basket.sum() / num_transactions
    freq_items = item_support[item_support >= min_support]

    # 2-item support
    pairs = {}
    items = freq_items.index.tolist()

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pair_support = (basket[items[i]] & basket[items[j]]).sum() / num_transactions
            if pair_support >= min_support:
                pairs[(items[i], items[j])] = pair_support

    return freq_items.to_dict(), pairs


def generate_rules(freq_items, freq_pairs, min_lift=1.0):
    """
    Generates association rules from frequent itemsets.
    """
    rules = []

    for (a, b), support_ab in freq_pairs.items():
        support_a = freq_items[a]
        support_b = freq_items[b]

        lift_ab = support_ab / (support_a * support_b)

        if lift_ab >= min_lift:
            rules.append({
                "antecedent": a,
                "consequent": b,
                "lift": lift_ab
            })
            rules.append({
                "antecedent": b,
                "consequent": a,
                "lift": lift_ab
            })

    return pd.DataFrame(rules)


def run_market_basket(df, user_col="user_id", item_col="item_id",
                      min_support=0.01, min_lift=1.0):
    """
    Full pipeline: basket → frequent itemsets → rules.
    """
    basket = prepare_transactions(df, user_col, item_col)
    freq_items, freq_pairs = fpgrowth(basket, min_support)
    rules = generate_rules(freq_items, freq_pairs, min_lift)
    return rules


def recommend_market_basket(item, rules_df, k=5):
    """
    Returns top associated items based on lift.
    """
    filtered = rules_df[rules_df["antecedent"] == item]

    if filtered.empty:
        return []

    return (
        filtered.sort_values("lift", ascending=False)
        .head(k)["consequent"]
        .tolist()
    )
