import numpy as np


class Kfold:
    def __init__(self, n_folds=10):
        self.n_folds = n_folds

    def split(self, X):
        """Return a list (of size n_folds) of tuples of the form (indexes for train set, indexes for test set)"""
        number_folds = self.n_folds
        data_size = len(X)
        len_ = int(data_size / number_folds)
        index_ = np.arange(data_size)
        split_indexes = []

        for i in range(number_folds):
            i1 = i * len_
            i2 = (i + 1) * len_
            test_indexes = index_[i1: i2]
            train_indexes = np.concatenate((index_[:i1], index_[i2:]))
            split_indexes.append((train_indexes, test_indexes))
        return split_indexes

    def run_once(self, model, X, y, train_index, test_index):
        X = X.to_numpy()
        model.fit(X[train_index], y[train_index])
        return model.predict(X[test_index])

    def cross_val(self, model, inp_score, X, y):
        '''
        Availiable parameters:
        - mean_squared_error
        - mean_absolute_error
        - accuracy_score
        - f1_score
        - precision_score
        - recall_score
        - roc_auc_score
        - max_error
        - r2_score
        '''
        number_folds = self.n_folds
        split_indexes = self.split(X)
        score = []
        for i in range(number_folds):
            train_index = split_indexes[i][0]
            test_index = split_indexes[i][1]
            y_pred = self.run_once(model, X, y, train_index, test_index)
            score.append(inp_score(y[test_index], y_pred))
        return np.mean(score)
