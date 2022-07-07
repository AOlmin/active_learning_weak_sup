from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

# from src.experiment_utils import parse_args, rbf_kernel, get_sine_regression_data, SineRegressionAnnotator
# from src.gp_model import GPModel
# from src.experiment_base_gp import main_experiment


def bald_acquisition(X_Pool, Queries, **kwargs):

    model = kwargs["model"]
    betas = kwargs["betas"]

    s = model.predict_model_var(X_Pool)
    assert s.shape == (X_Pool.shape[0], 1)

    # computing F_X
    var_y = model.var_y_given_f(X_Pool)
    Average_Entropy = np.sum(np.log(var_y), axis=-1)
    assert Average_Entropy.shape == (X_Pool.shape[0],) or np.isscalar(Average_Entropy)
    #0.5 * (d * (1 + np.log(2 * np.pi)) + np.log(var_y))
    Entropy_Average = np.sum(np.log(s + var_y), axis=-1)  #0.5 * (d * (1 + np.log(2 * np.pi)) + np.sum(np.log(s + var_y), axis=-1))
    assert Entropy_Average.shape == (X_Pool.shape[0],)

    U_X = Entropy_Average - Average_Entropy
    a_1d = U_X.flatten()
    x_pool_index = a_1d.argsort()[-Queries:][::-1]

    return x_pool_index, (np.ones((x_pool_index.shape[0])) * (-1)).astype(int)


# if __name__ == '__main__':
#
#
#     # Data params
#     rel_amp, fs, var = 0.2, 3, 0.01  # fs=7
#     prior_var = 0.01
#
#     store_file = "reg_data_rel_amp_" + str(rel_amp) + "_fs_" + str(fs) + "_var_" + str(var) + "_test_exp_" + str(0)
#
#     data = get_sine_regression_data(store_file=store_file, rel_amp=rel_amp, fs=fs, reuse_data=False, num_samples=10000,
#                                     non_uniform=True)
#
#     annotator = SineRegressionAnnotator(rel_amp, fs, prior_var, var)
#
#     args = parse_args()
#     kernel_params = np.ones((2,))
#     main_experiment(GPModel, rbf_kernel, kernel_params, data, prior_var, annotator.annotate,
#                     bald_acquisition, lambda a: a, lambda c: c, args,
#                     dir=args.save_dir + "BALD_test", e=0, ref_model=False)

