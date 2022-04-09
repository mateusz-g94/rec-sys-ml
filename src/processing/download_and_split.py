import tensorflow_datasets as tfds
from src.config.config import config


def download_data(path):
    # Ratings data.
    ratings = tfds.load(f"movielens/{config.model_config.data_version}-ratings", split="train", data_dir=path)
    # Features of all the available movies.
    movies = tfds.load(f"movielens/{config.model_config.data_version}-movies", split="train", data_dir=path)


if __name__=="__main__":
    download_data(path=config.model_config.data_dir_path)


