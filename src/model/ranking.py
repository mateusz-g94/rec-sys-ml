import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text
from src.config.config import config


class RankingModel(tf.keras.Model):

  def __init__(self, layer_sizes, unique_user_ids, unique_movie_titles):
    super().__init__()

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, config.model_config.model_params['ran']['user_embedding_dim'])
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, config.model_config.model_params['ran']['movie_embedding_dim'])
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(layer_sizes[0], activation="relu"),
      tf.keras.layers.Dense(layer_sizes[1], activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


class MovielensModel(tfrs.models.Model):

  def __init__(self, layer_sizes, unique_user_ids, unique_movie_titles):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel(layer_sizes, unique_user_ids, unique_movie_titles)
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model((features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")
    rating_predictions = self(features)
    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)