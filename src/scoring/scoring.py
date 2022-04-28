import numpy as np
import tensorflow as tf
from collections import Counter
import scann
import tensorflow_recommenders as tfrs


class RecSysInference(tf.keras.Model):

    def __init__(self, path_ret, path_ran, n_ran: int = 5):
        super().__init__()
        self.path_ret = path_ret
        self.path_ran = path_ran
        self.model_ret = tf.saved_model.load(self.path_ret)
        self.model_ran = tf.saved_model.load(self.path_ran)
        self.n_ran = n_ran

    def call(self, user_id: str):
        movies = self.model_ret([user_id])[1].numpy().flatten().tolist()
        ranking = {}
        for movie in movies:
            pred = self.model_ran({"user_id": tf.constant([user_id]), "movie_title": tf.constant([movie])})
            ranking[movie] = pred
        counter = Counter(ranking)
        ranking = counter.most_common(self.n_ran)
        return [movie[0] for movie in ranking], [movie[1].numpy().tolist()[0][0] for movie in ranking]