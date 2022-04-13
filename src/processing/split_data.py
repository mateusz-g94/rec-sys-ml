from src.config.config import config
import pandas as pd
import os


def f(s):
    s2 = pd.Series(0, index=s.index)
    s2.iloc[-1] = 1
    return s2


def split_data(path):

    ratings = pd.read_csv(path+'/v2/ratings.csv')
    movies = pd.read_csv(path+'/v2/movies.csv')
    ratings['flag_last'] = ratings.sort_values(by=['userId', 'timestamp']).groupby('userId')['timestamp'].apply(f)
    train = ratings[ratings.flag_last == 0]
    test = ratings[ratings.flag_last == 1]

    if not os.path.exists(path+'/v3'):
        os.mkdir(path+'/v3')

    train.to_csv(path+'/v3/ratings_train.csv')
    test.to_csv(path+'/v3/ratings_test.csv')
    movies.to_csv(path+'/v3/movies.csv')


if __name__ == "__main__":
    split_data(path=config.model_config.data_dir_path)