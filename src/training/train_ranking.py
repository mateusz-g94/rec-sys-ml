from src.model.ranking import MovielensModel
import numpy as np
import json
import tensorflow_recommenders as tfrs
from collections import OrderedDict
import tensorflow as tf
from src.config.config import config


if __name__ == '__main__':

    # Read data
    ratings_train = tf.data.experimental.CsvDataset(f'{config.model_config.data_dir_path}/v5/ratings_train.csv',
                                                    header = True,
                                                    record_defaults = [tf.string, tf.string, tf.float32])\
                                                        .map(lambda x1, x2, x3: OrderedDict({'user_id' : x1, 'movie_title' :x2, 'user_rating': x3}))

    ratings_test = tf.data.experimental.CsvDataset(f'{config.model_config.data_dir_path}/v5/ratings_test.csv',
                                                    header = True,
                                                    record_defaults = [tf.string, tf.string, tf.float32])\
                                                        .map(lambda x1, x2, x3: OrderedDict({'user_id' : x1, 'movie_title' :x2, 'user_rating': x3}))

    ratings_valid = tf.data.experimental.CsvDataset(f'{config.model_config.data_dir_path}/v5/ratings_valid.csv',
                                                    header = True,
                                                    record_defaults = [tf.string, tf.string, tf.float32])\
                                                        .map(lambda x1, x2, x3: OrderedDict({'user_id' : x1, 'movie_title' :x2, 'user_rating': x3}))

    # Create unique vocabulary
    unique_movie_titles = np.unique(np.concatenate(list(ratings_train.map(lambda x: x['movie_title']).batch(1000))))
    unique_user_ids = np.unique(np.concatenate(list(ratings_train.map(lambda x: x["user_id"]).batch(1000))))

    # Shuffle data
    ratings_train_cached = ratings_train.shuffle(10000).batch(2048).cache()
    ratings_test_cached = ratings_test.batch(4096).cache()
    ratings_valid_cached = ratings_valid.batch(4096).cache()

    # Train
    model = MovielensModel(
                    layer_sizes = [config.model_config.model_params['ran']['layer1'], config.model_config.model_params['ran']['layer2']],
                    unique_user_ids = unique_user_ids,
                    unique_movie_titles = unique_movie_titles)

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    history = model.fit(
        ratings_train_cached,
        validation_data=ratings_valid_cached,
        validation_freq=1,
        epochs=config.model_config.model_params['ran']['epochs'],
        verbose=1)

    # Results
    results = {}
    results['root_mean_squared_error'] = history.history['root_mean_squared_error']
    results['val_root_mean_squared_error'] = history.history['val_root_mean_squared_error']
    results['loss'] = history.history['loss']
    results['val_loss'] = history.history['val_loss']
    results['epochs'] = config.model_config.model_params['ret']['epochs']
    results['test_set'] = model.evaluate(ratings_test_cached, return_dict=True)
    with open(f'{config.model_config.results_dir_path}/results_ranking.json', 'w') as f:
        json.dump(results, f)

    # Save model
    tf.saved_model.save(model, f'{config.model_config.model_dir_path}/model_ran_pkl')