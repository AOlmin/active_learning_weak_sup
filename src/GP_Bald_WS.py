from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

# from src.experiment_utils import parse_args, rbf_kernel, get_sine_regression_data, SineRegressionAnnotator
# from src.gp_model import GPModel
# from src.experiment_base_gp import main_experiment


def bald_ws_acquisition(X_Pool, Queries, **kwargs):

    model = kwargs["model"]
    betas = kwargs["betas"]
    label_cost = kwargs["label_cost"]

    var_noisy_y = model.var_noisy_y_given_f(X_Pool.reshape((-1, 1, 1)), precision=betas.reshape((1, -1, 1)))
    #assert var_noisy_y.shape == (X_Pool.shape[0], betas.shape[0], 1)
    s = model.predict_model_var(X_Pool).reshape((X_Pool.shape[0], 1, -1))
    assert s.shape == (X_Pool.shape[0], 1, 1)

    d = var_noisy_y.shape[-1]
    assert d == 1
    Average_Entropy = np.sum(np.log(var_noisy_y), axis=-1)
    # Sum only relevant if multidim (assuming no correlation between dim)
    assert Average_Entropy.shape == (X_Pool.shape[0], betas.shape[0]) or Average_Entropy.shape == (1, betas.shape[0])
    Entropy_Average = np.sum(np.log(s + var_noisy_y), axis=-1)
    assert Entropy_Average.shape == (X_Pool.shape[0], betas.shape[0])

    U_X = Entropy_Average - Average_Entropy
    U_X /= label_cost(betas).reshape((1, -1))

    a_1d = U_X.flatten()

    inds = np.argpartition(a_1d, -Queries)[-Queries:]
    x_pool_index, beta_pool_index = np.unravel_index(inds[a_1d[inds].argsort()], U_X.shape)

    return x_pool_index, beta_pool_index


# if __name__ == '__main__':
#
#
#     # Data params
#     rel_amp, fs, var = 0.2, 3, 0.01  # fs=7
#     prior_var = 0.01
#
#     store_file = "reg_data_rel_amp_" + str(rel_amp) + "_fs_" + str(fs) + "_var_" + str(var) + "_test_exp_" + str(0)
#
#     data = get_sine_regression_data(store_file=store_file, rel_amp=rel_amp, fs=fs, reuse_data=True, num_samples=10000,
#                                     non_uniform=False)
#
#     annotator = SineRegressionAnnotator(rel_amp, fs, prior_var, var)
#
#     args = parse_args()
#     kernel_params = np.ones((2,))
#     main_experiment(GPModel, rbf_kernel, kernel_params, data, prior_var, annotator.annotate,
#                     bald_ws_acquisition, lambda a: a, lambda c: c, args,
#                     dir=args.save_dir + "BALD_WS_test", e=0, ref_model=False)







