from src.config.config import config
import pandas as pd


def test_data_layer2():
    ratings = pd.read_csv(config.model_config.data_dir_path+'/v2/ratings.csv', nrows=10)
    assert 'title' in ratings.columns.tolist()