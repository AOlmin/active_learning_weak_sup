from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings

import numpy as np
from scipy.stats import norm as scipy_norm

from src.experiment_utils import entropy
# from src.experiment_utils import parse_args, rbf_kernel, \
#     get_simple_class_data, SimpleClassAnnotator, acc_metric, entropy
# from src.gpc_model import GPClassifier
# from src.experiment_base_gp import main_experiment


def gpc_mi_ws_acquisition(X_Pool, Queries, **kwargs):

    model = kwargs["model"]
    alphas = kwargs["betas"]
    label_cost = kwargs["label_cost"]

    m, s = model.posterior_predict(X_Pool)
    assert m.shape == (X_Pool.shape[0], 1)
    assert s.shape == (X_Pool.shape[0], 1)
    m, s = m.reshape((X_Pool.shape[0], 1, -1)), s.reshape((X_Pool.shape[0], 1, -1))

    scaled_alphas = model.scale_precision(X_Pool, alphas).numpy().reshape((1, -1, 1))
    p = model.prob_noisy_y_given_prob_y(X_Pool.reshape((-1, 1, 1)), scipy_norm.cdf(m / np.sqrt(s + 1)),
                                        precision=alphas.reshape((1, -1, 1)))
    assert p.shape == (X_Pool.shape[0], alphas.shape[0], 1)
    Entropy_Average = entropy(p)
    assert Entropy_Average.shape == (X_Pool.shape[0], alphas.shape[0], 1)

    with warnings.catch_warnings():
        # TODO: FIX THIS
        warnings.filterwarnings("ignore", message="divide by zero encountered in log2")
        warnings.filterwarnings("ignore", message="invalid value encountered in log2")
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        alpha_h = entropy(scaled_alphas)

    alpha_h[0, -1, 0] = 0.0  # alpha_h[0, scaled_alphas.reshape(-1) == 1, 0] = 0.0
    assert scaled_alphas[0, -1, 0] == 1.0
    Average_Entropy = alpha_h
    assert Average_Entropy.shape == (1, alphas.shape[0], 1)

    U_X = (Entropy_Average - Average_Entropy)[:, :, 0]

    U_X /= label_cost(alphas).reshape((1, -1))
    a_1d = U_X.flatten()

    inds = np.argpartition(a_1d, -Queries)[-Queries:]
    x_pool_index, alpha_pool_index = np.unravel_index(inds[a_1d[inds].argsort()], U_X.shape)
    print(alpha_pool_index)

    return x_pool_index, alpha_pool_index


# if __name__ == '__main__':
#
#     # Data params
#     mean_1, mean_2 = -0.5 * np.ones((2,)), 0.5 * np.ones((2,))
#     store_file = "data/class_data_mean_1_" + str(mean_1[0]) + "_" + str(mean_1[1]) + "_mean_2_" + str(mean_2[0]) \
#                  + "_" + str(mean_2[1]) + "_test"
#
#     data = get_simple_class_data(store_file=store_file, mean_1=mean_1, mean_2=mean_2, reuse_data=True,
#                                  num_samples=10000)
#
#     annotator = SimpleClassAnnotator(mean_1, mean_2)
#
#     args = parse_args()
#     kernel_params = np.ones((2,))
#     q = 2
#     main_experiment(GPClassifier, rbf_kernel, kernel_params, data, 0.0, annotator.annotate,
#                     gpc_mi_ws_acquisition, lambda a: (2 * a - 1) ** q, lambda c: 0.5 * (c ** (1 / q) + 1), args,
#                     betas=np.linspace(0.8, 1, 100), metric=acc_metric, metric_name='ACC',
#                     dir=args.save_dir + "GPC_MI_WS_test", e=0, ref_model=False)



