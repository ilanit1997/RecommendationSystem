import abc
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class Recommender(abc.ABC):

    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()


    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        predict_fn = lambda row: self.predict(int(row['user']), int(row['item']), int(row['timestamp']))
        true_ratings['predicted'] = true_ratings.apply(predict_fn, axis=1)
        MSE = np.square(np.subtract(true_ratings['rating'], true_ratings['predicted'])).mean()
        RMSE = np.sqrt(MSE)
        return RMSE




class BaselineRecommender(Recommender):


    def initialize_predictor(self, ratings: pd.DataFrame): ##input:train
        self.ravg = ratings['rating'].mean()
        self.user_items = list(zip(ratings['user'], ratings['item']))
        self.b_users = ratings.groupby('user', as_index=False)['rating'].mean()
        self.b_items = ratings.groupby('item', as_index=False)['rating'].mean()


    def predict(self, user: int, item: int, timestamp: int) -> float: #input:test
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        self.bu = self.b_users[self.b_users['user'] == user]['rating'] - self.ravg
        self.bi = self.b_items[self.b_items['item'] == item]['rating'] - self.ravg
        predicted = self.ravg + self.bu.tolist()[0] + self.bi.tolist()[0]
        predicted = 0.5 if predicted < 0.5 else predicted
        predicted = 5 if predicted > 5 else predicted
        return predicted



class NeighborhoodRecommender(Recommender):

    def initialize_predictor(self, ratings: pd.DataFrame):
        #set variables for predictor
        self.ravg = ratings['rating'].mean()
        self.b_users = ratings.groupby('user', as_index=False)['rating'].mean()
        self.b_items = ratings.groupby('item', as_index=False)['rating'].mean()
        self.ratings = ratings.copy(deep=True)
        self.ratings['rating'] -= self.ravg
        self.users, self.items = list(set(self.ratings['user'])), list(set(self.ratings['item']))
        self.user_n, self.item_n  = len(self.users), len(self.items)
        self.sim = np.zeros((self.user_n, self.user_n))
        self.usersItem = {int(u): {int(i):r for (i,r) in zip(self.ratings[self.ratings['user']==u]['item'],
                                                   self.ratings[self.ratings['user']==u]['rating']) } for u in self.users}
        self.itemsUser = {int(i):{int(u):r for (u,r) in zip(self.ratings[self.ratings['item']==i]['user'],
                                                   self.ratings[self.ratings['item']==i]['rating']) } for i in self.items}

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        ##take users who rated item
        rel_user = self.itemsUser[item].keys()
        for user2 in rel_user:
            if user2!= user:
                self.user_similarity(user, user2)
        user, item = int(user), int(item)
        #take corr, index and rating of 3 most similar users
        current_sim = {u: (abs(self.sim[user][u]),self.sim[user][u]) for u in rel_user if abs(self.sim[user][u])!=0}
        current_sim = dict(sorted(current_sim.items(), key=lambda item: item[1][0], reverse=True)[0:3])
        best_corr = np.array([v[1] for v in list(current_sim.values())])
        best_ratings = np.array([self.usersItem[u][item] for u in current_sim.keys()])
        bu = self.b_users[self.b_users['user'] == user]['rating']  - self.ravg
        bi = self.b_items[self.b_items['item'] == item]['rating']  - self.ravg
        #predict rating for user
        predicted = self.ravg + bu.tolist()[0] + bi.tolist()[0] + (best_corr.T @ best_ratings)/\
                    sum(np.array([abs(i) for i in best_corr]))
        predicted = 0.5 if predicted < 0.5 else predicted
        predicted = 5 if predicted > 5 else predicted
        return predicted

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        i, j = int(user1), int(user2)
        norm1,norm2, inner = 0, 0, 0
        items1, items2 = self.usersItem[user1].keys(), self.usersItem[user2].keys()
        common_items = list(set(items1) & set(items2))
        if len(common_items) == 0:
            self.sim[i][j] = 0
            return self.sim[i][j]
        for item in common_items:
            r1, r2 = self.usersItem[user1][item], self.usersItem[user2][item]
            norm1+= r1**2
            norm2+= r2**2
            inner += r1*r2
        self.sim[i][j] = inner / (norm1 * norm2)**0.5
        return self.sim[i][j]




class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.ravg = ratings['rating'].mean()
        day_night = list(map(lambda x: 'day' if 6 <= int(datetime.fromtimestamp(x).hour) < 18
        else 'night', ratings['timestamp']))
        self.days = ['Friday', 'Saturday']
        Weekends = list(map(lambda x: 'Weekend' if datetime.fromtimestamp(x).strftime("%A") in self.days
        else 'Not_Weekend', ratings['timestamp']))
        self.ratings = ratings.copy(deep=True)
        self.ratings['Day_Night'] = day_night
        self.ratings['Weekend'] = Weekends
        self.A = pd.get_dummies(self.ratings, columns=['user', 'item', 'Day_Night', 'Weekend'])
        self.A.drop(columns=['Weekend_Not_Weekend', 'timestamp', 'rating'], axis=1, inplace=True)
        self.A = np.array(self.A)
        self.users = list(set(ratings['user']))
        self.num_users = len(self.users)
        self.items = list(set(ratings['item']))
        self.num_items = len(self.items)


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        bu = self.b[int(user)]
        bi = self.b[self.items.index(item) + self.num_users]
        predicted = self.ravg + bu + bi
        if 6 <= int(datetime.fromtimestamp(timestamp).hour) < 18:
            predicted += self.b[-3]
        else:
            predicted += self.b[-2]
        if datetime.fromtimestamp(timestamp).strftime("%A") in self.days:
            predicted += self.b[-1]

        predicted = 0.5 if predicted < 0.5 else predicted
        predicted = 5 if predicted > 5 else predicted
        return predicted


    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        self.y = np.array(self.ratings['rating']) - self.ratings['rating'].mean()
        self.b, __, __, __ = np.linalg.lstsq(self.A, self.y, rcond=None)
        return self.A, self.b, self.y



class CompetitionRecommender(Recommender):
    ## res : 0.9376942877954424
    ## run time: Took 188.16s

    def initialize_predictor(self, ratings: pd.DataFrame): ##input:train
        self.ravg = ratings['rating'].mean()
        self.user_items = list(zip(ratings['user'], ratings['item']))
        self.b_users = ratings.groupby('user', as_index=False)['rating'].mean()
        self.b_items = ratings.groupby('item', as_index=False)['rating'].mean()
        self.users = list(set(ratings['user']))
        self.items = list(set(ratings['item']))
        self.ravg = ratings['rating'].mean()
        #create weekend and day night
        day_night = list(map(lambda x: 'day' if 6 <= int(datetime.fromtimestamp(x).hour) < 18
        else 'night', ratings['timestamp']))
        self.days = ['Friday', 'Saturday']
        Weekends = list(map(lambda x: 'Weekend' if datetime.fromtimestamp(x).strftime("%A") in self.days
        else 'Not_Weekend', ratings['timestamp']))
        self.ratings = ratings
        self.ratings['Day_Night'] = day_night
        self.ratings['Weekend'] = Weekends
        b = self.ratings.groupby('Day_Night', as_index=False)['rating'].mean().to_numpy()
        self.b_D,  self.b_N= b[0][1]-self.ravg, b[1][1]-self.ravg
        b = self.ratings.groupby('Weekend', as_index=False)['rating'].mean().to_numpy()
        self.b_NotW , self.b_W= b[0][1]-self.ravg, b[1][1]-self.ravg


    def predict(self, user: int, item: int, timestamp: int) -> float: #input:test
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        b_h = self.b_D if 6 <= int(datetime.fromtimestamp(timestamp).hour) < 18 else self.b_N
        b_d = self.b_W if datetime.fromtimestamp(timestamp).strftime("%A") in self.days else self.b_NotW
        if item in self.items and user in self.users:
            self.bu, self.bi = self.b_users[self.b_users['user'] == user]['rating'] - self.ravg, \
                               self.b_items[self.b_items['item'] == item]['rating'] - self.ravg
            predicted = self.ravg + b_h + b_d + self.bu.tolist()[0] + self.bi.tolist()[0]
            predicted = 0.5 if predicted < 0.5 else predicted
            predicted = 5 if predicted > 5 else predicted
            return predicted
        elif user in self.users:
            self.bu = self.b_users[self.b_users['user'] == user]['rating'] - self.ravg
            predicted = self.ravg + b_h + b_d + self.bu.tolist()[0]
            predicted = 0.5 if predicted < 0.5 else predicted
            predicted = 5 if predicted > 5 else predicted
            return predicted
        elif item in self.items:
            self.bi = self.b_items[self.b_items['item'] == item]['rating'] - self.ravg
            predicted = self.ravg + b_h + b_d + self.bi.tolist()[0]
            predicted = 0.5 if predicted < 0.5 else predicted
            predicted = 5 if predicted > 5 else predicted
            return predicted
        else:
            predicted = self.ravg + b_d + b_h
            return predicted


