import tensorflow_datasets as tfds
from src.config.config import config
import pandas as pd

def test_data_layer1():
    ratings = pd.read_csv(config.model_config.data_dir_path+'/ratings.csv')
    if config.model_config.data_version == '1m':
        assert len(ratings) >= 1000000
    elif config.model_config.data_version == '20m':
        assert len(ratings) >= 20000000
    elif config.model_config.data_version == '100k':
        assert len(ratings) >= 100000
    else:
        raise Exception("Not appropiate data_version (1m, 20m, 100k).")
