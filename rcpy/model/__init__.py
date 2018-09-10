
import scipy
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Recommender(object):

    _MODEL_NAME = "Recommender"

    def __init__(self):
        print("init ---- Recommender")
        pass

    def get_model_name(self):
        return self._MODEL_NAME

    def recommend(self):
        raise NotImplementedError(
            str(type(self)) + " does not implement 'recommend'.")


class Popularity(Recommender):
    _MODEL_NAME = "popularity"

    def __init__(self, full_interactions_df, items_df=None):
        print("init ---- Popularity")

        Recommender.__init__(self)

        self.full_interactions_df = full_interactions_df
        self.items_df = items_df

    def get_popularity_df(self, content_column, score_column):
        popularity_df = self.full_interactions_df.groupby(content_column)[score_column]\
            .sum() \
            .sort_values(ascending=False) \
            .reset_index()
        return popularity_df

    def recommend(self, user_id, content_column, score_column, items_to_ignore=[], topn=10, full=False):

        popularity_df = self.get_popularity_df(content_column, score_column)
        recommendation_df = popularity_df[~ popularity_df[content_column].isin(items_to_ignore)] \
            .sort_values(score_column, ascending=False) \
            .head(topn)

        if full:
            if self.items_df is None:
                raise Exception("'items_df' is required in 'full' mode.")
            recommendation_df = recommendation_df.merge(self.items_df,
                                                        how="left",
                                                        left_on=content_column,
                                                        right_on=content_column)
        return recommendation_df


class ContentFilter(Recommender):

    _MODEL_NAME = "content-filter"

    def __init__(self, interactions_df, items_df, tfdif_matrix, content_column, score_column):
        print("init ---- Content")
        Recommender.__init__(self)

        self.interactions_df = interactions_df
        self.items_df = items_df
        self.tfdif_matrix = tfdif_matrix
        self.content_column = content_column
        self.score_column = score_column
        self.all_items_ids = self.items_df[content_column].tolist()

    def get_item_profile(self, item_id):
        index = self.all_items_ids.index(item_id)
        return self.tfdif_matrix[index:index+1]

    def get_user_items_profiles(self, user_item_ids):
        items_profiles_list =[self.get_item_profile(item_id) for item_id in user_item_ids]
        return scipy.sparse.vstack(items_profiles_list)

    def build_user_profile(self, user_id):
        user_interactions_df = self.interactions_df.loc[user_id]
        user_item_profiles = self.get_user_items_profiles(user_interactions_df[self.content_column])
        user_item_scores = np.array(user_interactions_df[self.score_column]).reshape(-1, 1)
        # weighted average of user item profiles by the interactions scores/strengths
        user_item_profiles_overall_score = np.sum(user_item_profiles.multiply(user_item_scores), axis=0) /\
                                   np.sum(user_item_scores)
        user_profile = sklearn.preprocessing.normalize(user_item_profiles_overall_score)
        return user_profile

    def build_users_profiles(self):
        users_profiles = dict()

        for user_id in self.interactions_df.index.unique().values:
            users_profiles.update({user_id: self.build_user_profile(user_id)})

        return users_profiles

    def get_items_similar_to_user_profile(self, user_profile, topn=1000):

        # get cosine similarities between user profile and the all user profiles
        similarities = cosine_similarity(user_profile, self.tfdif_matrix)
        # get top N similar items
        indicies = similarities.argsort().flatten()[-topn:]

        similar_items = sorted([(self.all_items_ids[i], similarities[0, i]) for i in indicies],
                               key=lambda x: -x[1])
        return similar_items

    def _get_score_by_content(self, user_id, content_column, score_column, items_to_ignore):

        user_profiles = self.build_users_profiles()

        similar_items = self.get_items_similar_to_user_profile(user_profiles[user_id])
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        return pd.DataFrame(similar_items_filtered,
                            columns=[content_column, score_column])

    def recommend(self, user_id, content_column, score_column, items_to_ignore=[], topn=10, full=False):

        recommendation_df = self._get_score_by_content(user_id,
                                                       content_column,
                                                       score_column,
                                                       items_to_ignore).head(topn)

        if full:
            if self.items_df is None:
                raise Exception("'items_df' is required in 'full' mode.")
            recommendation_df = recommendation_df.merge(self.items_df,
                                                        how="left",
                                                        left_on=content_column,
                                                        right_on=content_column)
        return recommendation_df


class CollaborativeFilter(Recommender):

    _MODEL_NAME = "collaborative-filter"

    def __init__(self,  scores_df, items_df=None):
        Recommender.__init__(self)

        self.scores_df = scores_df
        self.items_df = items_df

    def _get_score_by_collaborate(self, user_id, content_column, score_column, items_to_ignore):

        # get user scored ordered by descending score
        user_scores_ordered = self.scores_df[user_id] \
            .reset_index() \
            .rename(columns={user_id: score_column})

        user_scores_ordered = user_scores_ordered[~user_scores_ordered[content_column].isin(items_to_ignore)] \
            .sort_values(score_column, ascending=False)

        return user_scores_ordered

    def recommend(self, user_id, content_column, score_column, items_to_ignore=[], topn=10, full=False):

        recommendation_df = self._get_score_by_collaborate(user_id,
                                                           content_column,
                                                           score_column,
                                                           items_to_ignore).head(topn)
        # print(repr(recommendation_df))
        if full:
            if self.items_df is None:
                raise Exception("'items_df' is required in 'full' mode.")
            recommendation_df = recommendation_df.merge(self.items_df,
                                                        how="left",
                                                        left_on=content_column,
                                                        right_on=content_column)
        return recommendation_df


class Hybrid(ContentFilter, CollaborativeFilter):

    _MODEL_NAME = "Hybrid"

    def __init__(self, interactions_df, items_df, scores_df, tfdif_matrix, content_column, score_column):

        ContentFilter.__init__(self, interactions_df, items_df, tfdif_matrix, content_column, score_column)
        CollaborativeFilter.__init__(self, scores_df, items_df)

    def recommend(self, user_id, content_column, score_column, items_to_ignore=[], topn=10, full=False):

        content_recommendation_df = ContentFilter.recommend(
            self, user_id, content_column, score_column, items_to_ignore, int(1e16)) \
                .rename(columns={score_column: score_column+"CT"})
        collaborative_recommendation_df = CollaborativeFilter.recommend(
            self, user_id, content_column, score_column, items_to_ignore, int(1e16)) \
                .rename(columns={score_column: score_column+"CB"})


        recommendation_df = content_recommendation_df.reset_index().merge(
            collaborative_recommendation_df,
            how="inner",
            left_on=content_column,
            right_on=content_column,
        ).set_index("index")
        # print(repr(recommendation_df.head(20)))

        recommendation_df[score_column] = recommendation_df[score_column+"CT"] * recommendation_df[score_column+"CB"]
        recommendation_df = recommendation_df[[content_column, score_column]]
        recommendation_df = recommendation_df.sort_values(score_column, ascending=False).head(topn)

        if full:
            if self.items_df is None:
                raise Exception("'items_df' is required in 'full' mode.")
            recommendation_df = recommendation_df.merge(self.items_df,
                                                        how="left",
                                                        left_on=content_column,
                                                        right_on=content_column)
        return recommendation_df










