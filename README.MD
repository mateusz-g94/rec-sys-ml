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
4. Model evaluation
5. Package building locally
6. Package building AWS CodeArtefacts
7. CI/CD

### 1. Overview

The project shows how to build a <b>modern recommendation system</b> based on the architecture of neural networks.
<br><br>The goals of the project are:
- Train the recommendation engine
- Structure the project 
- Train the model
- Build the pypi package locally
- Build the pypi package in AWS CodeArtefacts
- Build CI/CD pipeline for building pypi package in the AWS 

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

### 4. Model evaluation

Retrieval model

![img.png](./results/ret.jpg)

Test metric:
* top 100 categorical accuracy = 7%

Ranking model

![img.png](./results/ran.jpg)

Test metric:
* RMSE = 0.98


