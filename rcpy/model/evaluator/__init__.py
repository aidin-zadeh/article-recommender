
import random
import pandas as pd


def is_in_hit_topn(item, recommended_items, topn):
    try:
        index = next(i for i, c in enumerate(recommended_items) if c == item)
    except:
        index = -1
    hit = int(index in range(0, topn))
    return hit, index


class RecallTopN(object):

    def __init__(
                self,
                full_interactions_df,
                test_interactions_df,
                train_interactions_df,
                items_df,
                content_column,
                score_column,
                sample_size=100):
            """

            :param full_interactions_df:
            :type full_interactions_df:
            :param test_interactions_df:
            :type test_interactions_df:
            :param train_interactions_df:
            :type train_interactions_df:
            :param all_items:
            :type all_items:
            :param sample_size:
            :type sample_size:
            """

            self.full_interactions_df = full_interactions_df
            self.test_interactions_df = test_interactions_df
            self.train_interactions_df = train_interactions_df
            self.all_items = set(items_df[content_column])
            self.content_column = content_column
            self.score_column = score_column
            self.sample_size = sample_size
            self.all_items = set(items_df[content_column])

    def evaluate(self, model):

        scores = []
        for i, user_id in enumerate(list(self.test_interactions_df.index.unique().values)):
            scores.append(self.evaluate_user(model, user_id, self.content_column, self.score_column))

        detailed_results_df = pd.DataFrame(scores).sort_values("nm_interacted_items", ascending=False)

        overall_recall_top5 = \
            detailed_results_df["nm_hits_top5"] \
                .sum() / float(detailed_results_df["nm_interacted_items"].sum())
        overall_recall_top10 = \
            detailed_results_df["nm_hits_top10"] \
                .sum() / float(detailed_results_df["nm_interacted_items"].sum())

        overall_results = pd.DataFrame(
            {"model_name": [model._MODEL_NAME],
             "recall_top5": [overall_recall_top5],
             "recall_top10": [overall_recall_top10]})
        return overall_results, detailed_results_df

    def evaluate_user(self, model, user_id, content_column, score_column):

        # get the items in the test set that user has interacted with
        user_test_interactions = self.test_interactions_df.loc[user_id]
        if type(user_test_interactions[content_column]) == pd.Series:
            user_test_interacted_items = set(user_test_interactions[content_column])
        else:
            user_test_interacted_items = set([int(user_test_interactions[content_column])])

        # user_test_items = self.get_interacted_items(user_id, self.content_column, self.test_interactions_df)
        nm_user_test_interactions = len(user_test_interacted_items)

        # get the items in train set that  user has interacted with
        user_train_interacted_items = self.get_interacted_items(user_id, content_column, self.train_interactions_df)


        # print(f"user={user_id}, content_column={content_column}, score_column={score_column}")
        # get user recommendations per unseen items
        user_recommendations_df = model.recommend(
            user_id,
            content_column,
            score_column,
            items_to_ignore=user_train_interacted_items,
            topn=int(1e16))
        nm_hits_top_5 = 0
        nm_hits_top_10 = 0

        for item in user_test_interacted_items:
            non_interacted_items_samples = self.get_non_interacted_items(
                interactions_df=self.full_interactions_df,
                user_id=user_id,
                content_column=content_column,
                sample_size=self.sample_size,
                seed=item % (2**32))

            filtered_items = non_interacted_items_samples.union(set([item]))
            # print(len(filtered_items))

            # get recommendations that are equal to either interacted item or are in the non-interacted items samples
            valid_user_recoms_df = user_recommendations_df[user_recommendations_df[content_column].isin(filtered_items)]

            hits_top_5, index_top_5 = is_in_hit_topn(item, valid_user_recoms_df[content_column].values, 5)
            nm_hits_top_5 += hits_top_5
            hits_top_10, index_top_10 = is_in_hit_topn(item, valid_user_recoms_df[content_column].values, 10)
            nm_hits_top_10 += hits_top_10

        recall_top5 = nm_hits_top_5 / float(nm_user_test_interactions)
        recall_top10 = nm_hits_top_10 / float(nm_user_test_interactions)

        print("user: ", "{:25d}".format(user_id),
              "\trecall@5: ", "{:0.3f}".format(recall_top5),
              "\trecall@10: ", "{:0.3f}".format(recall_top10))

        return {"user_id": user_id,
                "nm_hits_top5": nm_hits_top_5,
                "nm_hits_top10": nm_hits_top_10,
                "nm_interacted_items": nm_user_test_interactions,
                "recall_top5": recall_top5,
                "recall_top10": recall_top10}

    def get_non_interacted_items(self, interactions_df, user_id, content_column, sample_size, seed=42):

        # get all the items that users interacted with
        user_interacted_items = self.get_interacted_items(user_id, content_column, interactions_df)
        user_non_interacted_items = self.all_items - user_interacted_items
        # print(seed)
        random.seed(seed)
        return set(random.sample(user_non_interacted_items, sample_size))

    def get_interacted_items(self, user_id, content_column, interactions_df):
        interacted_items = interactions_df.loc[user_id][content_column]
        interacted_items = interactions_df if type(interacted_items) == pd.Series else [interacted_items]
        return set(interacted_items)
