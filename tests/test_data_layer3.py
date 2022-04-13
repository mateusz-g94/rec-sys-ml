from src.config.config import config
import pandas as pd


def test_data_layer3():
    ratings_train = pd.read_csv(config.model_config.data_dir_path+'/v3/ratings_train.csv')
    assert ratings_train.shape[0] >= 19861770
    ratings_test = pd.read_csv(config.model_config.data_dir_path + '/v3/ratings_test.csv')
    assert ratings_test.shape[0] >= 138493
