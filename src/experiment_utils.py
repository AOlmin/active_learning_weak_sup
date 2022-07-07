from pathlib import Path
import csv
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as scipy_mvn

import argparse


def get_sine_regression_data(store_file, rel_amp=1, fs=3, prior_var=0.01, var_factor=0.01, reuse_data=False,
                             num_samples=1000, non_uniform=False, split_data=True, heteroscedastic=False):

    data = None

    # Load data if it exists
    file = Path("data/" + store_file)
    if file.exists() and reuse_data:
        print("Loading data")
        with file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]

        data = np.array(tmp_raw_data, dtype=float)
        assert data.shape[0] == num_samples

    else:
        # Sample new data
        print("Sampling new data")

        file.parent.mkdir(parents=True, exist_ok=True)

        x = sine_regression_data_sample_input(num_samples, non_uniform=non_uniform)[:, np.newaxis]

        if heteroscedastic:
            y = sine_regression_data_sample_output(x, rel_amp, fs, prior_var, var_factor,
                                                   noise_function=heteroscedastic_noise)
        else:
            y = sine_regression_data_sample_output(x, rel_amp, fs, prior_var, var_factor, noise_function=constant_noise)

        # Combine and save data
        data = np.column_stack((x, y))
        np.random.shuffle(data)
        np.savetxt(file, data, delimiter=",")

    assert data.shape == (num_samples, 2)

    if split_data:
        start_pool = 10
        X_train, y_train = data[:start_pool, 0][:, np.newaxis], data[:start_pool, 1][:, np.newaxis]
        X_Pool, y_Pool = data[start_pool:int(0.6 * num_samples), 0][:, np.newaxis], \
                         data[start_pool:int(0.6 * num_samples), 1][:, np.newaxis]
        X_valid, y_valid = data[int(0.6 * num_samples):int(0.8 * num_samples), 0][:, np.newaxis], \
                           data[int(0.6 * num_samples):int(0.8 * num_samples), 1][:, np.newaxis]
        X_test, y_test = data[int(0.8 * num_samples):, 0][:, np.newaxis], data[int(0.8 * num_samples):, 1][:, np.newaxis]

        return X_train, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool
    else:
        return data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis]


def sine_regression_data_sample_input(num_samples, non_uniform=False):
    x_max = 5.0

    if non_uniform:
        prop = 0.9
        pos = np.random.binomial(num_samples, prop)
        x1 = np.random.uniform(low=0.0, high=x_max / 2, size=np.sum(pos))
        x2 = np.random.uniform(low=x_max / 2, high=x_max, size=num_samples - np.sum(pos))
        x = np.concatenate((x1, x2))

        assert x.shape[0] == num_samples
    else:
        x = np.random.uniform(low=0.0, high=x_max, size=num_samples)

    return x


def constant_noise(x, prior_var):
    return prior_var


def heteroscedastic_noise(x, prior_var):
    x_max = 5.0
    return prior_var * (1 + (x/x_max)**2)


def sine_regression_data_sample_output(x, rel_amp, fs, prior_var, var_factor, precision=0.0,
                                       noise_function=constant_noise):
    y = rel_amp * x * np.sin(fs * x) + np.sqrt(transform_noise(noise_function(x, prior_var), var_factor, precision)) \
                                               * np.random.normal(size=(x.shape[0], 1))

    return y


def get_toy_class_data(store_file, prior_var=0.0, var_factor=1.0, reuse_data=False, num_samples=1000, split_data=True,
                       version="v1"):

    assert version in ["v1", "v2", "v3"]

    data = None

    # Load data if it exists
    file = Path("data/" + store_file)
    if file.exists() and reuse_data:
        print("Loading data")
        with file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]

        data = np.array(tmp_raw_data, dtype=float)
        assert data.shape[0] == num_samples

    else:
        # Sample new data
        print("Sampling new data")

        file.parent.mkdir(parents=True, exist_ok=True)

        if split_data and version in ["v1", "v2"]:
            x1 = toy_class_data_sample_input(int(num_samples * 0.8), version=version)
            y1 = toy_class_data_sample_output(x1, prior_var, var_factor, version=version)
            x2, y2 = get_toy_class_test_data(int(num_samples * 0.2))

            data1 = np.column_stack((x1, y1))
            data2 = np.column_stack((x2, y2))
            np.random.shuffle(data1)
            np.random.shuffle(data2)

            data = np.row_stack((data1, data2))
        else:
            x = toy_class_data_sample_input(num_samples, version=version)
            y = toy_class_data_sample_output(x, prior_var, var_factor, version=version)

            # Combine data
            data = np.column_stack((x, y))
            np.random.shuffle(data)

        # Save data
        np.savetxt(file, data, delimiter=",")

    assert data.shape == (num_samples, 3)

    if split_data:
        start_pool = 5
        X_train, y_train = data[:start_pool, :-1], data[:start_pool, -1][:, np.newaxis]
        X_Pool, y_Pool = data[start_pool:int(0.6 * num_samples), :-1], \
                         data[start_pool:int(0.6 * num_samples), -1][:, np.newaxis]
        X_valid, y_valid = data[int(0.6 * num_samples):int(0.8 * num_samples), :-1], \
                           data[int(0.6 * num_samples):int(0.8 * num_samples), -1][:, np.newaxis]
        X_test, y_test = data[int(0.8 * num_samples):, :-1], \
                         data[int(0.8 * num_samples):, -1][:, np.newaxis]

        return X_train, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool

    else:
        return data[:, :-1], data[:, -1][:, np.newaxis]


def toy_class_data_sample_input(num_samples, version="v1"):

    if version == "v1":

        prob_block = 0.8
        num_block = np.random.binomial(n=num_samples, p=prob_block)

        # Sample block observations
        block_min = -0.3
        block_max = 0.3
        xb = np.random.uniform(low=block_min, high=block_max, size=(num_block, 2))

        # Sample non-block observations
        x_min = -2.0
        x_max = 2.0
        pb = (x_max - block_max) * (x_max - x_min)
        ps = (x_max - block_max) * (block_max - block_min)
        prob_comp = np.array([pb, pb, ps, ps]) / (2 * pb + 2 * ps)
        comp = np.argmax(np.random.multinomial(n=1, pvals=prob_comp, size=num_samples - num_block), axis=-1)

        x1 = np.random.uniform(low=np.array([x_min, x_min]), high=np.array([block_min, x_max]),
                               size=(np.sum(comp == 0), 2))
        x2 = np.random.uniform(low=np.array([block_max, x_min]), high=np.array([x_max, x_max]),
                               size=(np.sum(comp == 1), 2))
        x3 = np.random.uniform(low=np.array([block_min, block_max]), high=np.array([block_max, x_max]),
                               size=(np.sum(comp == 2), 2))
        x4 = np.random.uniform(low=np.array([block_min, x_min]), high=np.array([block_max, block_min]),
                               size=(np.sum(comp == 3), 2))

        x = np.row_stack((xb, x1, x2, x3, x4))

    elif version == "v2":

        prob_block = 0.8
        num_block = np.random.binomial(n=num_samples, p=prob_block)

        # Sample block observations
        block_min = -2.0
        block_max = -1.5
        xb = np.random.uniform(low=block_min, high=block_max, size=(num_block, 2))

        x_min = -2.0
        x_max = 2.0
        pb = (x_max - block_max) * (x_max - x_min)
        ps = (block_max - x_min) * (x_max - block_max)
        prob_comp = ps / (pb + ps)
        comp = np.random.binomial(n=1, p=prob_comp, size=num_samples - num_block)

        x1 = np.random.uniform(low=np.array([block_max, x_min]), high=np.array([x_max, x_max]),
                               size=(np.sum(comp == 0), 2))
        x2 = np.random.uniform(low=np.array([x_min, block_max]), high=np.array([block_max, x_max]),
                               size=(np.sum(comp == 1), 2))

        x = np.row_stack([xb, x1, x2])

    else:
        block_stops = np.array([-2.0, -1.0, 0.0, 1.0])
        block_mins_1, block_mins_2 = np.meshgrid(block_stops, block_stops)

        num_blocks = block_stops.shape[0]**2
        block_size = 0.5

        block_assign = np.argmax(np.random.multinomial(n=1, pvals=np.array([1 / num_blocks] * num_blocks),
                                                       size=num_samples), axis=-1)
        x = []
        for i, (block_min_1, block_min_2) in enumerate(zip(block_mins_1.reshape(-1), block_mins_2.reshape(-1))):
            new_x = np.random.uniform(low=np.array([block_min_1, block_min_2]),
                                      high=np.array([block_min_1 + block_size, block_min_2 + block_size]),
                                      size=(np.sum(block_assign == i), 2))
            if i == 0:
                x = new_x
            else:
                x = np.row_stack((x, new_x))

    return x


def toy_class_data_sample_output(x, prior_var, var_factor, precision=np.array([[1.0]]), version="v1"):

    scaled_prec = transform_noise(prior_var, var_factor, precision)

    if version == "v1":
        block_min = -0.3
        block_max = 0.3

        # Check if in block
        in_block = (x[:, 0] >= block_min) * (x[:, 0] <= block_max) * (x[:, 1] >= block_min) * (x[:, 1] <= block_max)

        prob_block = 0.6
        prob_y = (x[:, 0] <= 0).astype(np.float64)[:, np.newaxis]
        prob_y[in_block] = prob_block

    elif version == "v2":
        # Check if in block
        block_min = -2.0
        block_max = -1.5
        in_block = (x[:, 0] >= block_min) * (x[:, 0] <= block_max) * (x[:, 1] >= block_min) * (x[:, 1] <= block_max)

        prob_block = 1.0
        prob_y = (x[:, 0] <= 0).astype(np.float64)[:, np.newaxis]
        prob_y[in_block] = prob_block

    else:
        diff = (x[:, 0] - x[:, 1])
        step = 0.5
        prob_y = ((diff >= -step) * (diff <= step) + (diff >= 3 * step) * (diff <= 5 * step) \
                  + (diff <= -3 * step) * (diff >= -5 * step))[:, np.newaxis]

    prob_noise = (2 * scaled_prec - 1) * prob_y + 1 - scaled_prec
    y = np.random.binomial(n=1, p=prob_noise, size=(x.shape[0], 1)).astype(np.float64)
    y[y == 0.0] = -1.0

    return y


def get_simple_class_data(store_file, mean_1=-np.ones((2,)), mean_2=np.ones((2,)), reuse_data=False,
                          num_samples=1000,  non_uniform=False, split_data=True):

    data = None

    # Load data if it exists
    file = Path("data/" + store_file)
    if file.exists() and reuse_data:
        print("Loading data")
        with file.open(newline="") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",", quotechar="|")
            tmp_raw_data = [data for data in csv_reader]

        data = np.array(tmp_raw_data, dtype=float)
        assert data.shape[0] == num_samples

    else:
        # Sample new data
        print("Sampling new data")

        file.parent.mkdir(parents=True, exist_ok=True)

        x = simple_class_data_sample_input(num_samples, mean_1, mean_2, non_uniform=non_uniform)
        y = simple_class_data_sample_output(x, mean_1, mean_2, non_uniform=non_uniform)

        # Combine and save data
        data = np.column_stack((x, y))
        np.random.shuffle(data)
        np.savetxt(file, data, delimiter=",")

    assert data.shape == (num_samples, 3)

    if split_data:
        start_pool = 10
        X_train, y_train = data[:start_pool, :-1], data[:start_pool, -1][:, np.newaxis]
        X_Pool, y_Pool = data[start_pool:int(0.6 * num_samples), :-1], \
                         data[start_pool:int(0.6 * num_samples), -1][:, np.newaxis]
        X_valid, y_valid = data[int(0.6 * num_samples):int(0.8 * num_samples), :-1], \
                           data[int(0.6 * num_samples):int(0.8 * num_samples), -1][:, np.newaxis]
        X_test, y_test = data[int(0.8 * num_samples):, :-1], \
                         data[int(0.8 * num_samples):, -1][:, np.newaxis]

        return X_train, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool
    else:
        return data[:, :-1], data[:, -1][:, np.newaxis]


def simple_class_data_sample_input(num_samples, mean_1, mean_2, non_uniform=False):

    cov = np.eye(2)

    if non_uniform:
        prop = 0.9
        pos = np.random.binomial(num_samples, prop)
        x1 = np.random.multivariate_normal(mean=mean_1, cov=cov, size=np.sum(pos))
        x2 = np.random.multivariate_normal(mean=mean_2, cov=cov, size=num_samples - np.sum(pos))

    else:
        x1 = np.random.multivariate_normal(mean=mean_1, cov=cov, size=int(np.ceil(num_samples / 2)))
        x2 = np.random.multivariate_normal(mean=mean_2, cov=cov, size=int(np.floor(num_samples / 2)))

    return np.concatenate((x1, x2))


def simple_class_data_sample_output(x, mean_1, mean_2, precision=1.0, non_uniform=False):

    if non_uniform:
        prop = 0.9  # probability of negative class
    else:
        prop = 0.5

    x_prob_1 = scipy_mvn.pdf(x, mean=mean_1, cov=np.eye(2))
    x_prob_2 = scipy_mvn.pdf(x, mean=mean_2, cov=np.eye(2))

    y_prob = x_prob_1 * prop / (x_prob_1 * prop + x_prob_2 * (1 - prop))
    yt_prob = precision * y_prob + (1 - precision) * (1 - y_prob)

    y = np.random.binomial(1, yt_prob).astype(np.float64)
    y[y==0] = -1.0

    return y


def transform_noise(prior_var, var_factor, precision):
    return prior_var + scale_noise(var_factor, precision)


def scale_noise(var_factor, precision):
    return var_factor * precision


def get_toy_class_test_data(num_samples):
    """For version 1 and 2"""

    x_min = -2.0
    x_max = 2.0
    x = np.random.uniform(low=x_min, high=x_max, size=(num_samples, 2))

    y = 2.0 * (x[:, 0] <= 0) - 1.0

    return x, y[:, np.newaxis]


def round_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def rbf_kernel(x_n, x_m, params):
    """ SE/RBF kernel """
    return (params[0]**2) * tf.math.exp(- tf.math.reduce_euclidean_norm(x_n - x_m, axis=-1)**2 / (2 * params[1]**2))


def laplacian_kernel(x_n, x_m, params):
    """ SE/RBF kernel """
    return tf.math.exp(- tf.norm(x_n - x_m, ord=1, axis=-1) / params)


def mse_metric(preds, y_true):
    m, _ = preds

    return ((m - y_true)**2).mean()


def acc_metric(preds, y_true):
    y_pred = np.rint(preds)
    y_pred[y_pred == 0] = -1

    return np.mean(y_pred == y_true)


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2(1 - p)


def standardise_data(data, mean, std):
    return (data - mean) / std


def identity_function(x):
    return x


def identify_unique(x):
    x_unique = np.unique(x)
    # For now we only use this for single queries
    assert x_unique.shape[0] == 1

    return x_unique[:, np.newaxis]


def parse_args():
    """Arg parser"""
    parser = argparse.ArgumentParser(description="EXPERIMENT_ARGUMENTS")

    al_group = parser.add_argument_group('al_group')
    al_group.add_argument("--nb_experiments", type=int,
                          default=1,
                          help="Number of times to repeat experiment")
    al_group.add_argument("--nb_epochs", type=int,
                          default=100,
                          help="Number of epochs used for training")
    al_group.add_argument("--nb_queries", type=int,
                          default=1,
                          help="Number of queries per iteration")
    al_group.add_argument("--nb_acquisitions", type=int,
                          default=3,  # 196
                          help="Number of times to look for label")
    al_group.add_argument("--nb_dropout_iterations", type=int,
                          default=5,
                          help="Number of dropout iterations")
    al_group.add_argument("--save_dir",
                          default="res_test/",
                          help="Folder for saving results")
    regr_group = parser.add_argument_group('regr_group')
    regr_group.add_argument("--q", type=float,
                            default=1,
                            help="Order of cost function")
    regr_group.add_argument("--fs", type=float,
                            default=3,
                            help="Frequency of sine curve")
    regr_group.add_argument("--fit",
                            action='store_true',
                            help="Whether to run experiments or not (otherwise only evaluation will be done)")
    regr_group.add_argument("--fit_hyp",
                            action='store_true',
                            help="Whether to fit hyperparameters or not")
    regr_group.add_argument("--interpolate",
                            action='store_true',
                            help="Whether to interpolate results for visualisation or not")
    class_group = parser.add_argument_group('class_group')
    class_group.add_argument("--version",
                             default="v1",
                             help="Class toy example version (v1/v2/v3)")
    uci_group = parser.add_argument_group("mat_group")
    uci_group.add_argument("--data_dir",
                           default="data/",
                           help="Directory of material data.")

    return parser.parse_args()


class SineRegressionAnnotator:

    def __init__(self, rel_amp, fs, prior_var, var_factor, heteroscedastic=False):

        self.rel_amp = rel_amp
        self.fs = fs
        self.prior_var = prior_var
        self.var_factor = var_factor

        if heteroscedastic:
            self.noise_func = heteroscedastic_noise
        else:
            self.noise_func = constant_noise

    def annotate(self, x, y, precision):
        return sine_regression_data_sample_output(x, self.rel_amp, self.fs, self.prior_var, self.var_factor, precision,
                                                  noise_function=self.noise_func)


class ToyClassAnnotator:

    def __init__(self, prior_var, var_factor, version="v1"):

        assert version in ["v1", "v2", "v3"]

        self.version = version
        self.prior_var = prior_var
        self.var_factor = var_factor

    def annotate(self, x, y, precision):
        return toy_class_data_sample_output(x, self.prior_var, self.var_factor, precision, version=self.version)


