import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderEngine:
    def __init__(self, df):
        # Ensure column consistency
        self.df = df.rename(columns={df.columns[0]: 'user', df.columns[1]: 'item', df.columns[2]: 'rating'})
        self.pivot = self.df.pivot_table(index='user', columns='item', values='rating')
        self.user_sim_df = None

    def build_similarity(self):
        # Fill NA with 0 for similarity math
        filled_matrix = self.pivot.fillna(0)
        sim_matrix = cosine_similarity(filled_matrix)
        self.user_sim_df = pd.DataFrame(sim_matrix, index=self.pivot.index, columns=self.pivot.index)

    def get_user_recommendations(self, target_user, n=5):
        if target_user not in self.pivot.index:
            return self.get_popular_items(n)

        # Get top 10 similar users
        sim_users = self.user_sim_df[target_user].sort_values(ascending=False)[1:11]
        
        # Weighted average of neighbor ratings
        neighbor_ratings = self.pivot.loc[sim_users.index]
        weights = sim_users.values
        
        # Dot product to get scores
        scores = np.dot(weights, neighbor_ratings.fillna(0))
        recommendations = pd.Series(scores, index=self.pivot.columns)
        
        # Filter out what they already rated
        already_rated = self.pivot.loc[target_user].dropna().index
        recommendations = recommendations.drop(already_rated)
        
        return recommendations.sort_values(ascending=False).head(n)

    def get_popular_items(self, n=5):
        # Fallback: Items with highest average rating and at least 2 ratings
        popular = self.df.groupby('item')['rating'].agg(['mean', 'count'])
        return popular[popular['count'] >= 2].sort_values('mean', ascending=False).head(n)['mean']
