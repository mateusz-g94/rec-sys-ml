from src.config.config import config
import wget
import os
import zipfile
import shutil


def download_data(path):

    url = f"https://files.grouplens.org/datasets/movielens/ml-{config.model_config.data_version}.zip"
    if not os.path.exists(path+'/v1'):
        os.mkdir(path+'/v1')
    wget.download(url=url, out=path+'/v1')
    with zipfile.ZipFile(path + f"/v1/ml-{config.model_config.data_version}.zip", "r") as zip_ref:
        zip_ref.extractall(path)
    os.remove(path + f"/v1/ml-{config.model_config.data_version}.zip")
    files = os.listdir(path+'/ml-20m')
    for file in files:
        shutil.move(path+'/ml-20m/'+file, path + '/v1/' + file)
    os.rmdir(path+'/ml-20m')


if __name__ == "__main__":
    download_data(path=config.model_config.data_dir_path)


