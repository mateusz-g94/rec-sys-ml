Topics
------
- Data Science
- Machine Learning Engineering
- MLOps


Table of Contents
--------
1. Overview
2. Data
3. Model architecture
4. Model training and locally package building
5. Model evaluation
   1. Iteration I
6. Package building AWS CodeArtefacts
7. CI/CD

### 1. Overview

The project shows how to build a <b>modern recommendation system</b> based on the architecture of neural networks.
<br><br>The goals of the project are:
- Train the recommendation engine
- Structure the project 
- Train the model
- Build the PyPi package locally
- Build the PyPi package in AWS CodeArtefacts
- Build CI/CD pipeline for building PyPi package in the AWS 

This is the first of two parts. The second part of the project will be in a separate repository and will concern building a <b>Flask REST API</b> that will share predictions from the model. 

### 2. Data

Data source: https://grouplens.org/datasets/movielens/20m/

The data includes information on 5-star ratings derived from the movie recommendation system

Important datasets:
* ratings.csv columns: userId,movieId,rating,timestamp
* movies.csv columns: movieId,title,genres
 
### 3. Model architecture

Model consists of two stages:
* Retrieval Stage - efficiency select hundreds of candidates from millions of candidates
  * ./src/model/retrieval.py
  * ./src/training/train_retrieval.py
* Ranking Stage - predict user ranking for hundreds of candidates and order by scoring
  * ./src/model/ranking.py
  * ./src/training/train_ranking.py
  
[![](https://mermaid.ink/img/pako:eNpV0D0PgjAQBuC_0tykCZK4Mpj4OemCI2W40BMaaUtKS6LAf7coJtrp8t7TvE17KIwgSKC02FTsnHLNwtlmF9NJYoWxjW9ztlptBiVraXQ7sN0iJWcldVizq8OSlp9Lu4mxofJaWBIB7hcp6rvU5R_bv9lhbljnv-kxi-P4NxmEedJUepq9fm8hAkVWoRTh6f2UcHAVKeKQhFGgvXPgegzONwIdHYV0xkJyw7qlCNA7c33oAhJnPX3RQWL4BjWr8QWBvVqg)](https://mermaid.live/edit#pako:eNpV0D0PgjAQBuC_0tykCZK4Mpj4OemCI2W40BMaaUtKS6LAf7coJtrp8t7TvE17KIwgSKC02FTsnHLNwtlmF9NJYoWxjW9ztlptBiVraXQ7sN0iJWcldVizq8OSlp9Lu4mxofJaWBIB7hcp6rvU5R_bv9lhbljnv-kxi-P4NxmEedJUepq9fm8hAkVWoRTh6f2UcHAVKeKQhFGgvXPgegzONwIdHYV0xkJyw7qlCNA7c33oAhJnPX3RQWL4BjWr8QWBvVqg)

### 4. Model training and package building locally

Check config.yml file specification and make sure the training parameters are set properly.

Download data, prepare data, train both models and test:
```commandline
tox
```

Train one model:
```commandline
tox -e train_retrieval 
tox -e train_ranking
```

Build package locally:
```commandline
python3 -m pip install --upgrade build
python3 -m build
```

Install package to local PyPi repository:
```commandline
pip install ./dist/recsysmodel-0.0.1-py3-none-any.whl
```

Check package:
```python
from recsysmodel.scoring.predict import predict
user_id = 20
# Recomendations for user 20
rec = predict(20)
print(rec)

Results:
[b'Wallace & Gromit: A Close Shave (1995)', b'Gattaca (1997)', b'Thing, The (1982)', b'Cube (1997)', b'Black Orchid, The (1958)'], [4.351129531860352, 4.121507167816162, 4.089163303375244, 3.755620241165161, 3.7031068801879883])
```

### 5. Model evaluation

* <b>Iteration I</b> <br>
Assumptions:
  * 5% of observations
  * retrieval ::
    epochs: 10
    user_embedding_dim: 32
    title_embedding_dim: 25
    title_text_embedding_dim: 7
    layer1: 16
    layer2: 8
    scann_k: 15

  * ranking ::
    epochs: 10
    user_embedding_dim: 32
    movie_embedding_dim: 32
    layer1: 32
    layer2: 16

Retrieval model

![img.png](./results/ret.jpg)

Test metric:
* top 100 categorical accuracy = 7%

Ranking model

![img.png](./results/ran.jpg)

Test metric:
* RMSE = 0.98


