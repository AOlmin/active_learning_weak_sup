from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf

from src.experiment_utils import rbf_kernel, get_sine_regression_data, SineRegressionAnnotator, \
    heteroscedastic_noise, parse_args
from src.gp_model import GPModel, GPModelII
from src.experiment_base_gp import main_experiment
from experiment.res_plot import vis_res_main

from src.Random import random_acquisition
from src.GP_Bald import bald_acquisition
from src.GP_Bald_WS import bald_ws_acquisition
from src.GP_MI_WS import mi_ws_acquisition


def sine_reg_exp_main(args, rel_amp, fs, prior_var, var_factor, suff, cost_fun, precision_fun, non_uniform=False,
                      input_independent_noise=False, fit=True, kernel_params=np.ones((2,)),
                      fit_hyp=False, interpolate_results=False, include_mi=False):

    # Check if running on GPU
    tf.config.list_physical_devices('GPU')

    heteroscedastic = True
    kernel_params[1] = 3 / fs

    annotator = SineRegressionAnnotator(rel_amp, fs, prior_var, var_factor, heteroscedastic=heteroscedastic)

    acquisition_functions = [random_acquisition, bald_acquisition, bald_ws_acquisition]

    refs = ["Random", "Bald", "Bald_WS"]

    if include_mi:
        acquisition_functions += [mi_ws_acquisition]
        refs += ["MI_WS"]

    if fit:
        for e in range(args.nb_experiments):
            print(e)
            store_file = "reg_data_rel_amp_" + str(rel_amp) + "_fs_" + str(fs) + "_var_" + str(prior_var) + "_exp_" \
                         + str(e) + suff

            data = get_sine_regression_data(store_file=store_file, rel_amp=rel_amp, fs=fs, prior_var=prior_var,
                                            var_factor=var_factor, reuse_data=True,
                                            num_samples=10000, non_uniform=non_uniform, heteroscedastic=heteroscedastic)

            if non_uniform:
                # Test on uniform data
                X_train, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool = data

                X_test_u, y_test_u = get_sine_regression_data(store_file=store_file + "_test", rel_amp=rel_amp, fs=fs,
                                                              prior_var=prior_var, var_factor=var_factor,
                                                              reuse_data=True, num_samples=X_test.shape[0],
                                                              non_uniform=False, split_data=False,
                                                              heteroscedastic=heteroscedastic)

                data = X_train, y_train, X_valid, y_valid, X_test_u, y_test_u, X_Pool, y_Pool

            for ref, acq_fun in zip(refs, acquisition_functions):

                if input_independent_noise:
                    model_class = GPModelII
                else:
                    model_class = GPModel

                main_experiment(model_class, rbf_kernel, kernel_params, data, prior_var, annotator.annotate,
                                acq_fun, cost_fun, precision_fun, args, var_factor=var_factor,
                                betas=1 - np.linspace(0.0, 1.0, 100), dir=args.save_dir + ref + suff, e=e,
                                fit_hyperparams=fit_hyp, noise_func=heteroscedastic_noise)

    vis_res_main(args, suff, interpolate_results=interpolate_results, include_mi=include_mi)


if __name__ == '__main__':
    rel_amp, fs = 0.2, 3
    prior_var, var_factor = 0.01, 0.01
    suff = "_fs_3_a_test"

    args = parse_args()

    sine_reg_exp_main(args, rel_amp, fs, prior_var, var_factor, suff, lambda a: 0.1 + (1-a)**0.2,
                      lambda c: (c-0.1)**(1/0.2))
