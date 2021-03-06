import json
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs

from recsysmodel.config.config import config
from recsysmodel.model.retrieval import MovielensModel

if __name__ == "__main__":

    # Read data
    ratings_train = tf.data.experimental.CsvDataset(
        f"{config.model_config.data_dir_path}/v5/ratings_train.csv",
        header=True,
        record_defaults=[tf.string, tf.string, tf.float32],
    ).map(lambda x1, x2, x3: OrderedDict({"user_id": x1, "movie_title": x2}))

    ratings_test = tf.data.experimental.CsvDataset(
        f"{config.model_config.data_dir_path}/v5/ratings_test.csv",
        header=True,
        record_defaults=[tf.string, tf.string, tf.float32],
    ).map(lambda x1, x2, x3: OrderedDict({"user_id": x1, "movie_title": x2}))

    ratings_valid = tf.data.experimental.CsvDataset(
        f"{config.model_config.data_dir_path}/v5/ratings_valid.csv",
        header=True,
        record_defaults=[tf.string, tf.string, tf.float32],
    ).map(lambda x1, x2, x3: OrderedDict({"user_id": x1, "movie_title": x2}))

    movies = tf.data.experimental.CsvDataset(
        f"{config.model_config.data_dir_path}/v5/movies.csv",
        header=True,
        record_defaults=[
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
            tf.int32,
            tf.string,
            tf.string,
        ],
    ).map(lambda x1, x2, x3, x4, x5, x6, x7: OrderedDict({"movie_title": x6}))

    # Create unique vocabulary
    unique_movie_titles = np.unique(
        np.concatenate(list(movies.map(lambda x: x["movie_title"]).batch(1000)))
    )
    unique_user_ids = np.unique(
        np.concatenate(list(ratings_train.map(lambda x: x["user_id"]).batch(1000)))
    )

    # Shuffle data
    ratings_train_cached = ratings_train.shuffle(10000).batch(2048).cache()
    ratings_test_cached = ratings_test.batch(4096).cache()
    ratings_valid_cached = ratings_valid.batch(4096).cache()

    # Train
    model = MovielensModel(
        layer_sizes=[
            config.model_config.model_params["ret"]["layer1"],
            config.model_config.model_params["ret"]["layer2"],
        ],
        movies=movies.map(lambda x: x["movie_title"]),
        unique_user_ids=unique_user_ids,
        unique_movie_titles=unique_movie_titles,
    )

    model.compile(optimizer=tf.keras.optimizers.Adam())

    history = model.fit(
        ratings_train_cached,
        validation_data=ratings_valid_cached,
        validation_freq=1,
        epochs=config.model_config.model_params["ret"]["epochs"],
        verbose=1,
    )

    # Results
    results = {}
    results["factorized_top_k/top_100_categorical_accuracy"] = history.history[
        "factorized_top_k/top_100_categorical_accuracy"
    ]
    results["val_factorized_top_k/top_100_categorical_accuracy"] = history.history[
        "val_factorized_top_k/top_100_categorical_accuracy"
    ]
    results["loss"] = history.history["loss"]
    results["val_loss"] = history.history["val_loss"]
    results["epochs"] = config.model_config.model_params["ret"]["epochs"]
    results["arch"] = [
        config.model_config.model_params["ret"]["layer1"],
        config.model_config.model_params["ret"]["layer2"],
    ]
    results["scann_k"] = config.model_config.model_params["ret"]["scann_k"]
    results["test_set"] = model.evaluate(ratings_test_cached, return_dict=True)
    with open(
        f"{config.model_config.results_dir_path}/results_retrieval.json", "w"
    ) as f:
        json.dump(results, f)

    # Create scan index object
    scann_index = tfrs.layers.factorized_top_k.ScaNN(
        model.query_model, k=config.model_config.model_params["ret"]["scann_k"]
    )
    scann_index.index_from_dataset(
        tf.data.Dataset.zip(
            (
                movies.map(lambda x: x["movie_title"]).batch(100),
                movies.map(lambda x: x["movie_title"])
                .batch(100)
                .map(model.candidate_model),
            )
        )
    )

    # Save model
    _, names = scann_index(tf.constant(["22"]))

    tf.saved_model.save(
        scann_index, f"{config.model_config.model_dir_path}/model_ret_pkl"
    )
