from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from scipy.stats import norm as scipy_norm

from src.experiment_utils import entropy
# from src.experiment_utils import parse_args, rbf_kernel, get_simple_class_data, SimpleClassAnnotator, acc_metric
# from src.gpc_model import GPClassifier
# from src.experiment_base_gp import main_experiment


def gpc_bald_acquisition(X_Pool, Queries, **kwargs):
    model = kwargs["model"]

    m, s = model.posterior_predict(X_Pool)
    assert m.shape == (X_Pool.shape[0], 1)
    assert s.shape == (X_Pool.shape[0], 1)

    p = scipy_norm.cdf(m / np.sqrt(s + 1))
    assert p.shape == (X_Pool.shape[0], 1)
    Entropy_Average = entropy(p)
    c = np.sqrt(np.pi * np.log(2) / 2)
    Average_Entropy = (c / np.sqrt(s + c**2)) * np.exp(- m**2 / (2 * (s + c**2)))
    assert Entropy_Average.shape == (X_Pool.shape[0], 1)
    assert Average_Entropy.shape == (X_Pool.shape[0], 1)

    U_X = Entropy_Average - Average_Entropy
    a_1d = U_X.flatten()
    x_pool_index = a_1d.argsort()[-Queries:][::-1]

    return x_pool_index, (np.ones((x_pool_index.shape[0])) * (-1)).astype(int)


# if __name__ == '__main__':
#
#
#     # Data params
#     mean_1, mean_2 = -0.5 * np.ones((2,)), 0.5 * np.ones((2,))
#     store_file = "data/class_data_mean_1_" + str(mean_1[0]) + "_" + str(mean_1[1]) + "_mean_2_" + str(mean_2[0]) \
#                  + "_" + str(mean_2[1]) + "_test"
#
#     data = get_simple_class_data(store_file=store_file, mean_1=mean_1, mean_2=mean_2, reuse_data=True,
#                                 num_samples=10000)
#
#     annotator = SimpleClassAnnotator(mean_1, mean_2)
#
#     args = parse_args()
#     kernel_params = np.ones((2,))
#     main_experiment(GPClassifier, rbf_kernel, kernel_params, data, 0.0, annotator.annotate,
#                     gpc_bald_acquisition, lambda a: a, lambda c: c, args, metric=acc_metric, metric_name='ACC',
#                     dir=args.save_dir + "GPC_BALD_test", e=0, ref_model=False)

