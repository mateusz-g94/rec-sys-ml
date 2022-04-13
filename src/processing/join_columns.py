from src.config.config import config
import pandas as pd
import os


def join_columns(path):

    ratings = pd.read_csv(path+'/v1/ratings.csv')
    movies = pd.read_csv(path+'/v1/movies.csv')
    ratings_new = ratings.merge(movies[['movieId', 'title']], on = 'movieId')

    if not os.path.exists(path+'/v2'):
        os.mkdir(path+'/v2')

    ratings_new.to_csv(path+'/v2/ratings.csv')
    movies.to_csv(path+'/v2/movies.csv')


if __name__ == "__main__":
    join_columns(path=config.model_config.data_dir_path)