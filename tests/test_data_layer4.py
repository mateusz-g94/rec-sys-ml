from src.config.config import config
import pandas as pd
import tensorflow as tf
from collections import OrderedDict


def test_data_layer4():
    ratings_train = pd.read_csv(config.model_config.data_dir_path+'/v4/ratings_train.csv', nrows=10)
    assert len(ratings_train.columns) == 3
    ratings_test = pd.read_csv(config.model_config.data_dir_path + '/v4/ratings_test.csv', nrows=10)
    assert len(ratings_test.columns) == 3
    ratings_train2 = tf.data.experimental.make_csv_dataset(config.model_config.data_dir_path+'/v4/ratings_train.csv', \
                                                           batch_size=2048)
    assert isinstance(ratings_train2.element_spec, OrderedDict)
    assert ratings_train2.element_spec['user_id'].shape[0] == 2048