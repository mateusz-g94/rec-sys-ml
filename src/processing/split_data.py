from src.config.config import config
import pandas as pd
import os
import shutil


def f(s):
    s2 = pd.Series(0, index=s.index)
    s2.iloc[-1] = 1
    return s2


def split_data(path):

    ratings = pd.read_csv(path+'/v3/ratings.csv')
    movies = pd.read_csv(path+'/v3/movies.csv')
    ratings['flag_last'] = ratings.sort_values(by=['userId', 'timestamp']).groupby('userId')['timestamp'].apply(f)
    train = ratings[ratings.flag_last == 0]
    test = ratings[ratings.flag_last == 1]
    valid_users = test[['userId']].drop_duplicates().sample(frac=config.preprocessing_config.validation_freq)
    valid = test[test.userId.isin(valid_users.userId)]
    test = test[~test.userId.isin(valid_users.userId)]

    if os.path.exists(path + '/v4'):
        shutil.rmtree(path + '/v4/')
    os.mkdir(path + '/v4')

    train.to_csv(path+'/v4/ratings_train.csv')
    valid.to_csv(path+'/v4/ratings_valid.csv')
    test.to_csv(path+'/v4/ratings_test.csv')
    movies.to_csv(path+'/v4/movies.csv')


if __name__ == "__main__":
    split_data(path=config.model_config.data_dir_path)