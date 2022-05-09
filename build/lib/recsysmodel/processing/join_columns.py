import os
import shutil

import pandas as pd

from recsysmodel.config.config import config


def join_columns(path):

    ratings = pd.read_csv(path + "/v2/ratings.csv")
    movies = pd.read_csv(path + "/v2/movies.csv")
    ratings_new = ratings.merge(movies[["movieId", "title"]], on="movieId")

    if os.path.exists(path + "/v3"):
        shutil.rmtree(path + "/v3/")
    os.mkdir(path + "/v3")

    ratings_new.to_csv(path + "/v3/ratings.csv")
    movies.to_csv(path + "/v3/movies.csv")


if __name__ == "__main__":
    join_columns(path=config.model_config.data_dir_path)
