import inspect
from tqdm import tqdm
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, max_error, mean_squared_error, \
    r2_score


def grid_search(classifier,
                parameters,
                score=accuracy_score,
                x_test_inp=X_test,
                y_test_inp=y_test,
                x_train_inp=X_train,
                y_train_inp=y_train):
    '''
    This implementation of a grid search is for 4 hyperparameters
    Available scores for the function:
        - accuracy_score,
        - f1_score,
        - precision_score,
        - recall_score,
        - max_error,
        - mean_squared_error,
        - r2_score
    Returns: score and dict of parameters as keys with their best values as values
    '''
    best_score = 0
    values_lists = list(parameters.values())
    key_list = list(parameters.keys())

    #   Checks if all entered parameters and score are exist in parameters
    #   of the classifier and list if scores in requirements
    try:
        all(item in inspect.getfullargspec(classifier.__init__).args
            for item in list(params_dict.keys()))
        raise SystemExit(
            'entered parameters do not match parameters of the classifier')
    except:

        try:
            score not in (accuracy_score, f1_score, precision_score,
                          recall_score, max_error, mean_squared_error,
                          r2_score)
            raise SystemExit('entered score does not match requirements')
        except:

            for params in tqdm(product(*values_lists)):
                params_dict = {
                    key_list[0]: params[0],
                    key_list[1]: params[1],
                    key_list[2]: params[2],
                    key_list[3]: params[3]
                }

                clf = classifier(**params_dict)
                clf.fit(x_train_inp, y_train_inp)
                y_pred = clf.predict(x_test_inp)
                score_inp = score(y_test_inp, y_pred)

                if score_inp > best_score:
                    best_params = params
                    best_score = score_inp
            return best_score, dict(zip(key_list, best_params))
