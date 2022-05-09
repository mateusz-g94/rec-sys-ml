import json

import matplotlib.pyplot as plt

from recsysmodel.config.config import config


def plot_metric(js, model):

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    if model == "ran":
        metric = "root_mean_squared_error"
    elif model == "ret":
        metric = "factorized_top_k/top_100_categorical_accuracy"

    ax[0].plot(js[metric], label="train")
    ax[0].plot(js[f"val_{metric}"], label="val")
    ax[0].hlines(
        js["test_set"][metric],
        xmin=0,
        xmax=len(js[metric]),
        colors="g",
        linestyle="--",
        label="test",
    )
    ax[0].set_title(metric)

    ax[1].plot(js["loss"], label="train")
    ax[1].plot(js["val_loss"], label="val")
    ax[1].hlines(
        js["test_set"]["loss"],
        xmin=0,
        xmax=len(js[metric]),
        colors="g",
        linestyle="--",
        label="test",
    )
    ax[1].set_title("loss")

    plt.legend()
    plt.savefig(f"./results/{model}.jpg")


def eval_metrics(model):

    if model == "ran":
        with open(
            f"{config.model_config.results_dir_path}/results_ranking.json", "r"
        ) as f:
            results = json.load(f)

    elif model == "ret":
        with open(
            f"{config.model_config.results_dir_path}/results_retrieval.json", "r"
        ) as f:
            results = json.load(f)

    plot_metric(js=results, model=model)


if __name__ == "__main__":

    eval_metrics("ran")
    eval_metrics("ret")
