from src.scoring.scoring import RecSysInference
from src.config.config import config


def predict(user_id):

    model = RecSysInference(f'{config.model_config.model_dir_path}/model_ret_pkl',
                            f'{config.model_config.model_dir_path}/model_ran_pkl')

    return model(str(user_id))


if __name__ == '__main__':

    print(predict(15))