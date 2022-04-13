from src.config.config import config
import pandas as pd
import os


def select_columns(path):

    ratings_train = pd.read_csv(path+'/v3/ratings_train.csv')
    ratings_test = pd.read_csv(path+'/v3/ratings_test.csv')
    movies = pd.read_csv(path+'/v3/movies.csv')

    ratings_train = ratings_train[['userId', 'title', 'rating']]
    ratings_train = ratings_train.rename(columns={'userId' : 'user_id', 'title': 'movie_title', 'rating': 'user_rating'})

    ratings_test = ratings_test[['userId', 'title', 'rating']]
    ratings_test = ratings_test.rename(columns={'userId': 'user_id', 'title': 'movie_title', 'rating': 'user_rating'})

    if not os.path.exists(path+'/v4'):
        os.mkdir(path+'/v4')

    ratings_train.to_csv(path+'/v4/ratings_train.csv', index=False)
    ratings_test.to_csv(path+'/v4/ratings_test.csv', index=False)
    movies.to_csv(path+'/v4/movies.csv')


if __name__ == "__main__":
    select_columns(path=config.model_config.data_dir_path)