

from random import *
import pandas as pd
import math

col_names = ['user_id','repo_id','created_at']
reader = pd.read_csv('../dataset/user_repo.csv',names=col_names,nrows= 10000)

M = 1; # The time of trials


# Split the data into Training Set and Test Set（0 <= k <= M - 1）
def SplitData(data, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for user_id, repo_id in data:
        if random.randint(0, M) == k:
            test.append([user_id, repo_id])
        else:
            train.append([user_id, repo_id])
    return train, test


def ItemSimilarity(train):
    # calculate co-rated users between items
    C = dict()
    N = dict()
    for user_id, repo_id in train.items():
        for i in users:
            N[i] += 1
            for j in users:
                if i == j:
                    continue
                C[i][j] += 1

    # calculate finial similarity matrix W

    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W[u][v] = cij / math.sqrt(N[i] * N[j])
    return W


def Recommendation(train, user_id, W, K):
    rank = dict()
    ru = train[user_id]
    for i,pi in ru.items():
        for j, wj in sorted(W[i].items(),key=itemgetter(1), reverse=True)[0:K]:
            if j in ru:
                continue
            rank[j] += pi * wj
    return rank