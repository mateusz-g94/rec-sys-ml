from src.config.config import config
import pandas as pd


def test_data_layer3():
    ratings = pd.read_csv(config.model_config.data_dir_path+'/v3/ratings.csv', nrows=10)
    assert 'title' in ratings.columns.tolist()