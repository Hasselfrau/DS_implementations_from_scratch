import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class kNNClassifier(BaseEstimator):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.data = X
        self.labels = y
        return X, y

    def predict(self, X):
        self.y_pred = list()

        for row in X:
            output = self.distance(row - self.data)
            self.y_pred.append(self.pred(output))

        return self.y_pred

    def distance(self, row):
        return np.sqrt(np.sum(np.square(row), axis=1))

    def pred(self, output):
        idx = pd.DataFrame({'norm': list(output)}).sort_values('norm').iloc[:self.n_neighbors].index

        return pd.DataFrame(self.labels[list(idx)])[0].value_counts().sort_values(ascending=False).index[0]

    def score(self, X, y):
        y_pred = self.predict(X)
        error = (y_pred != y).sum()
        return 100 * error / len(y)
