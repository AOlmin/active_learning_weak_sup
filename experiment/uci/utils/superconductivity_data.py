import csv
import numpy as np

from src.experiment_utils import scale_noise


def get_superconductivity_data(store_file="../data/superconductivity.csv", split_data=True):

    with open(store_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        raw_data = [l for i, l in enumerate(csv_reader)]

    #print(raw_data[0])
    data = np.array(raw_data[1:]).astype(np.float64)
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1][:, np.newaxis]

    if split_data:
        start_pool = 20
        num_samples = X.shape[0]
        X_train, y_train = X[:start_pool, :], y[:start_pool, :]
        X_Pool, y_Pool = X[start_pool:int(0.8 * num_samples), :], y[start_pool:int(0.8 * num_samples), :]
        X_test, y_test = X[int(0.8 * num_samples):, :], y[int(0.8 * num_samples):, :]
        X_valid, y_valid = X_train.copy(), y_train.copy() # Just placeholders, never used

        assert y_train.shape[0] + y_Pool.shape[0] + y_test.shape[0] == num_samples

        return X_train, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool

    else:
        return X, y


def superconductivity_data_sample_output(y, precision, var_factor=1.0):
    y_tilde = y + np.sqrt(scale_noise(var_factor, precision)) * np.random.normal(size=(y.shape[0], 1))
    return y_tilde


class SuperconductivityAnnotator:
    def __init__(self, var_factor=1.0):
        self.var_factor = var_factor

    def annotate(self, x, y, precision):
        return superconductivity_data_sample_output(y, precision, self.var_factor)


if __name__ == '__main__':
    get_superconductivity_data()


