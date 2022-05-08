import pandas as pd
import scann
import tensorflow as tf

from recsysmodel.config.config import config


def test_serving_ret():
    model = tf.saved_model.load(f'{config.model_config.model_dir_path}/model_ret_pkl')
    _, names_pred = model(['22'])
    rec = names_pred[0].numpy().flatten().tolist()
    assert len(rec) == 15
