from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf

from src.experiment_utils import parse_args, rbf_kernel, get_toy_class_data, ToyClassAnnotator, acc_metric
from src.gpc_model import GPClassifier
from src.experiment_base_gp import main_experiment
from experiment.res_plot import vis_res_main

from src.Random import random_acquisition
from src.GPC_Bald import gpc_bald_acquisition
from src.GPC_Bald_WS import gpc_bald_ws_acquisition
from src.GPC_MI_WS import gpc_mi_ws_acquisition


def toy_class_exp_main(args, suff, cost_fun, precision_fun, prior_var=0.8, var_factor=0.2, version="v1", fit=True,
                       kernel_params=np.ones((2,)), fit_hyp=False, interpolate_results=False, include_mi=False):

    # Check if running on GPU
    tf.config.list_physical_devices('GPU')

    annotator = ToyClassAnnotator(prior_var, var_factor, version=version)

    acquisition_functions = [random_acquisition, gpc_bald_acquisition, gpc_bald_ws_acquisition]
    refs = ["Random", "Bald", "Bald_WS"]

    if include_mi:
        acquisition_functions += [gpc_mi_ws_acquisition]
        refs += ["MI_WS"]

    if fit:
        for e in range(args.nb_experiments):
            store_file = "toy_class_data_" + version + "_exp_" + str(e) + suff

            data = get_toy_class_data(store_file=store_file, prior_var=prior_var, var_factor=var_factor,
                                      reuse_data=True, num_samples=10000, version=version)

            for ref, acq_fun in zip(refs, acquisition_functions):

                main_experiment(GPClassifier, rbf_kernel, kernel_params, data, prior_var, annotator.annotate,
                                acq_fun, cost_fun, precision_fun, args, var_factor=var_factor,
                                betas=np.linspace(0.0, 1.0, 100), metric=acc_metric, metric_name="ACC",
                                dir=args.save_dir + ref + suff, e=e, fit_hyperparams=fit_hyp)

    vis_res_main(args, suff, metric_name="ACC", interpolate_results=interpolate_results, include_mi=include_mi,
                 group="toy_class", log_scale=False)


if __name__ == '__main__':

    args = parse_args()
    suff = "_toy_test"

    toy_class_exp_main(args, suff, lambda a: a, lambda c: c)

