import numpy as np
from math import sqrt

class SelfAttention:
    def __init__(self):
        self.wk = np.array([
            [0.65, -0.34, 0.56],
            [0.23, 0.54, -0.12],
            [-0.43, 0.67, 0.42]
        ])

        self.wv = np.array([
            [0.75, -0.23, 0.34],
            [0.76, 0.23, -0.56],
            [-0.11, 0.33, 0.32]
        ])

        self.wq = np.array([
            [0.18, -0.42, 0.91],
            [0.55, 0.07, -0.31],
            [-0.66, 0.84, 0.29]
        ])

        self.dk = 3

    def _get_query(self, embeddings):
        queries = embeddings @ self.wq
        return np.array(queries)

    def _get_key(self, embeddings):
        keys = embeddings @ self.wk
        return np.array(keys)

    def _get_value(self, embeddings):
        values = embeddings @ self.wv
        return np.array(values)

    def _get_scores(self, queries, keys):
        scores = (queries @ keys.T) / sqrt(self.dk)
        return np.array(scores)

    def _softmax(self, scores):
        attention = []
        for row in scores:
            exp_score = np.exp(row)
            exp_sum = np.sum(exp_score)
            exp_score = exp_score / exp_sum
            attention.append(exp_score)
        return np.array(attention)






