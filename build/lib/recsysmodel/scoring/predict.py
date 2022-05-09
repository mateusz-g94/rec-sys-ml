from recsysmodel.config.config import config
from recsysmodel.scoring.scoring import RecSysInference
from recsysmodel.utils.decorators import timer


@timer
def predict(user_id):

    model = RecSysInference(
        f"{config.model_config.model_dir_path}/model_ret_pkl",
        f"{config.model_config.model_dir_path}/model_ran_pkl",
    )

    return model(str(user_id))


if __name__ == "__main__":

    print(predict(15))
