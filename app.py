def get_top_n_recommendations_sparse(user_idx, data_dict, nn_model, n=5, top_k_neighbors=50):
    """
    Fast user-based CF:
    1. Find K nearest neighbors
    2. Aggregate all neighbor ratings (vectorized)
    3. Remove items already rated by target user
    4. Return top-N recommendations
    """

    rating_csr = data_dict['rating_csr']

    # --- 1. Find nearest neighbors (only once) ---
    k_query = min(top_k_neighbors + 1, rating_csr.shape[0])
    distances, neighbors = nn_model.kneighbors(
        rating_csr[user_idx],
        n_neighbors=k_query,
        return_distance=True
    )
    distances = distances.ravel()
    neighbors = neighbors.ravel()

    # remove self from neighbors
    mask = neighbors != user_idx
    neighbors = neighbors[mask]
    distances = distances[mask]

    if len(neighbors) == 0:
        return []

    # convert cosine distance -> similarity
    sims = 1 - distances
    sims = np.maximum(sims, 0)

    # --- 2. Vectorized weighted scoring across ALL products ---
    neighbor_ratings = rating_csr[neighbors]  # shape: (K, items)
    weights = sims.reshape(-1, 1)             # shape: (K, 1)

    # weighted sum of ratings for all items
    weighted_sum = neighbor_ratings.multiply(weights).sum(axis=0).A1

    # total similarity weight for each item
    weight_total = neighbor_ratings.sign().multiply(weights).sum(axis=0).A1

    preds = np.zeros_like(weight_total)
    mask_valid = weight_total > 0
    preds[mask_valid] = weighted_sum[mask_valid] / weight_total[mask_valid]

    # --- 3. Remove items already rated by user ---
    user_rated = rating_csr[user_idx].toarray().ravel() > 0
    preds[user_rated] = -1  # mark invalid

    # --- 4. Top N recommendations ---
    top_idx = np.argsort(preds)[-n:][::-1]
    top_idx = [i for i in top_idx if preds[i] > 0]

    idx_to_prod = data_dict['idx_to_prod']
    return [(idx_to_prod[i], float(preds[i])) for i in top_idx]
