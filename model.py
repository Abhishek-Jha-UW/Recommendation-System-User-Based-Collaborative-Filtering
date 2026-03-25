import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderEngine:
    def __init__(self, df):
        # Dynamically rename the first three columns to ensure consistency
        self.df = df.copy()
        self.df.columns = ['user', 'item', 'rating']
        # Create the user-item matrix
        self.pivot = self.df.pivot_table(index='user', columns='item', values='rating')

    def get_user_based(self, target_user, n=5):
        """USER-BASED: 'Users like you also bought...'"""
        if target_user not in self.pivot.index:
            return pd.Series(dtype=float)
        
        # Calculate User Similarity Matrix
        sim_matrix = cosine_similarity(self.pivot.fillna(0))
        user_sim_df = pd.DataFrame(sim_matrix, index=self.pivot.index, columns=self.pivot.index)
        
        # Get top 10 similar users
        similar_users = user_sim_df[target_user].sort_values(ascending=False)[1:11]
        
        # Weighted average of their ratings
        neighbor_ratings = self.pivot.loc[similar_users.index].fillna(0)
        weights = similar_users.values
        scores = np.dot(weights, neighbor_ratings)
        
        recs = pd.Series(scores, index=self.pivot.columns)
        # Drop items the user has already seen
        already_rated = self.pivot.loc[target_user].dropna().index
        return recs.drop(already_rated, errors='ignore').sort_values(ascending=False).head(n)

    def get_item_based(self, target_user, n=5):
        """ITEM-BASED: 'Because you liked X, you might like Y...'"""
        if target_user not in self.pivot.index:
            return pd.Series(dtype=float)
        
        user_history = self.pivot.loc[target_user].dropna()
        if user_history.empty:
            return pd.Series(dtype=float)

        # Calculate Item Similarity Matrix (Transpose the pivot)
        item_sim_matrix = cosine_similarity(self.pivot.T.fillna(0))
        item_sim_df = pd.DataFrame(item_sim_matrix, index=self.pivot.columns, columns=self.pivot.columns)
        
        # Score items based on similarity to the user's rated items
        scores = item_sim_df[user_history.index].dot(user_history)
        return scores.drop(user_history.index, errors='ignore').sort_values(ascending=False).head(n)

    def get_market_basket(self, target_user, n=5):
        """MARKET BASKET: 'People frequently bought these together...'"""
        if target_user not in self.pivot.index:
            return pd.Series(dtype=float)

        # Convert to binary (1 = purchased, 0 = not)
        basket = self.pivot.notna().astype(int)
        user_items = basket.loc[target_user]
        items_bought = user_items[user_items == 1].index
        
        if items_bought.empty:
            return pd.Series(dtype=float)

        # Item Co-occurrence (how often items appear in the same basket)
        co_matrix = basket.T.dot(basket)
        # Sum co-occurrences for all items in user's history
        scores = co_matrix[items_bought].sum(axis=1)
        return scores.drop(items_bought, errors='ignore').sort_values(ascending=False).head(n)
