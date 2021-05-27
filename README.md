# RecommendationSystem

Background:

The data file is ratings.csv. Every record in the file is of the form user, item, rating, timestamp.
user – The user’s unique identifier
item – The item’s unique identifier
rating – The rating that was given to the item by the user, it is in the range [0.5,5]
timestamp – The timestamp in which the rating was given.


**Baseline model**


The first baseline model for recommender systems is 
𝑟𝑢𝑖 ̂ = 𝑅̂ + 𝑏𝑢 + 𝑏𝑖 where
𝑅̂ - average of all the ratings in the user-item ratings matrix 𝑅,
𝑏𝑢 - average rating for user 𝑢 and
𝑏𝑖 -  average rating for item 𝑖.


**Collaborative filtering**


NeighborhoodRecommender makes use of the similarities of users and the similarities of items to
make predictions. 
We will be using only the user similarities.
The prediction is done with the 3 nearest neighbors.


**Regression model**

The rating estimate for user 𝑢, item 𝑖 and timestamp 𝑡 is:

𝑟𝑢𝑖𝑡 = 𝑅̂ + 𝑏𝑢 + 𝑏𝑖 + 𝑏𝑑 + 𝑏𝑛 + 𝑏i.

where:
𝑏𝑑 – A parameter for ratings that were given in daytime (between 6am and 6pm).
𝑏𝑛 – A parameter for ratings that were given in the night (between 6pm and 6am).
𝑏𝑤 – A parameter for ratings that were given in the weekend (Friday or Saturday).


This is a least squares problem:
min 𝑏𝑢,𝑏𝑖,𝑏𝑑,𝑏𝑛,𝑏𝑤 ‖𝑋𝛽 − 𝑦‖2


To solve the least squares problem, we use np.linalg.lstsq.

**CompetitionRecommender**


We tried to find the lowest RMSE score on the ratings_comp data (800,000 ratings), by combining ls parameters and baseline prediction.

