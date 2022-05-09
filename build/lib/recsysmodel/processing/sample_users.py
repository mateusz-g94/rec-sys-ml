import os
import shutil

import pandas as pd

from recsysmodel.config.config import config


def sample_users(path):

    ratings = pd.read_csv(path + "/v1/ratings.csv")
    users = (
        ratings[["userId"]]
        .drop_duplicates()
        .sample(frac=config.preprocessing_config.user_freq)
    )
    movies = pd.read_csv(path + "/v1/movies.csv")
    ratings_new = ratings[ratings.userId.isin(users.userId)]

    if os.path.exists(path + "/v2"):
        shutil.rmtree(path + "/v2/")
    os.mkdir(path + "/v2")

    ratings_new.to_csv(path + "/v2/ratings.csv")
    movies.to_csv(path + "/v2/movies.csv")


if __name__ == "__main__":
    sample_users(path=config.model_config.data_dir_path)
