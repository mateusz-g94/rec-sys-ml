import os
import shutil

import pandas as pd

from recsysmodel.config.config import config


def select_columns(path):

    ratings_train = pd.read_csv(path + "/v4/ratings_train.csv")
    ratings_test = pd.read_csv(path + "/v4/ratings_test.csv")
    ratings_valid = pd.read_csv(path + "/v4/ratings_valid.csv")
    movies = pd.read_csv(path + "/v4/movies.csv")

    ratings_train = ratings_train[["userId", "title", "rating"]]
    ratings_train = ratings_train.rename(
        columns={"userId": "user_id", "title": "movie_title", "rating": "user_rating"}
    )

    ratings_test = ratings_test[["userId", "title", "rating"]]
    ratings_test = ratings_test.rename(
        columns={"userId": "user_id", "title": "movie_title", "rating": "user_rating"}
    )

    ratings_valid = ratings_valid[["userId", "title", "rating"]]
    ratings_valid = ratings_valid.rename(
        columns={"userId": "user_id", "title": "movie_title", "rating": "user_rating"}
    )

    if os.path.exists(path + "/v5"):
        shutil.rmtree(path + "/v5/")
    os.mkdir(path + "/v5")

    ratings_train.to_csv(path + "/v5/ratings_train.csv", index=False)
    ratings_test.to_csv(path + "/v5/ratings_test.csv", index=False)
    ratings_valid.to_csv(path + "/v5/ratings_valid.csv", index=False)
    movies.to_csv(path + "/v5/movies.csv")


if __name__ == "__main__":
    select_columns(path=config.model_config.data_dir_path)
