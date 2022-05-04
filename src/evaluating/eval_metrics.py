from src.config.config import config
import json


def eval_metrics(model):

    if model == 'ran':
        with open(f'{config.model_config.results_dir_path}/results_ranking.json', 'r') as f:
           results = json.load(f)

    elif model == 'ret':
        with open(f'{config.model_config.results_dir_path}/results_retrieval.json', 'r') as f:
           results = json.load(f)

    if model == 'all':
        # with open(f'{config.model_config.results_dir_path}/results_ranking.json', 'r') as f:
        #    results = json.load(f)
        return {}

    return dict(results)


if __name__ == '__main__':

    print(eval_metrics('ret'))