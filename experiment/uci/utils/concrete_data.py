import numpy as np
import pandas as pd

from src.experiment_utils import scale_noise

def get_concrete_data(store_file="../data/concrete.xlsx", split_data=True):

    df = pd.read_excel(store_file, dtype=np.float64)

    X = np.asarray(df.iloc[:, :-1])
    y = np.asarray(df["Concrete compressive strength"])

    num_samples = X.shape[0]
    shuffled_ind = np.random.permutation(num_samples)

    X, y = X[shuffled_ind, :], y[shuffled_ind, np.newaxis]

    if split_data:
        start_pool = 20
        X_train, y_train = X[:start_pool, :], y[:start_pool, :]
        X_Pool, y_Pool = X[start_pool:int(0.8 * num_samples), :], y[start_pool:int(0.8 * num_samples), :]
        X_test, y_test = X[int(0.8 * num_samples):, :], y[int(0.8 * num_samples):, :]
        X_valid, y_valid = X_train.copy(), y_train.copy() # Just placeholders, never used

        assert y_train.shape[0] + y_Pool.shape[0] + y_test.shape[0] == num_samples

        return X_train, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool

    else:
        return X, y


def concrete_data_sample_output(y, precision, var_factor=1.0):
    y_tilde = y + np.sqrt(scale_noise(var_factor, precision)) * np.random.normal(size=(y.shape[0], 1))
    return y_tilde


class ConcreteAnnotator:

    def __init__(self, var_factor=1.0):
        self.var_factor = var_factor

    def annotate(self, x, y, precision):
        return concrete_data_sample_output(y, precision, self.var_factor)


if __name__ == '__main__':
    get_concrete_data()


