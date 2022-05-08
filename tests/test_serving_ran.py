import pandas as pd
import scann
import tensorflow as tf

from recsysmodel.config.config import config


def test_serving_ran():
    model = tf.saved_model.load(f'{config.model_config.model_dir_path}/model_ran_pkl')
    pred = model({"user_id": tf.constant(['1']), "movie_title": tf.constant(['Evita (1996)'])})
    assert isinstance(pred.numpy().tolist()[0][0], float)
