from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf

from src.experiment_utils import parse_args, rbf_kernel
from src.experiment_base_gp import main_experiment
from experiment.res_plot import vis_res_main

from src.gp_model import GPModel
from src.Random import random_acquisition
from src.GP_Bald import bald_acquisition
from src.GP_Bald_WS import bald_ws_acquisition
from src.GP_MI_WS import mi_ws_acquisition


def uci_exp_main(args, store_file, data_loader, annotator_class, prior_var, suff, cost_fun, precision_fun, var_factor=1.0,
                 fit=True, kernel_params=np.ones((2,)), fit_hyp=False, interpolate_results=False, include_mi=False):

    # Check if running on GPU
    tf.config.list_physical_devices('GPU')

    acquisition_functions = [random_acquisition, bald_acquisition, bald_ws_acquisition]
    annotator = annotator_class()

    refs = ["Random", "Bald", "Bald_WS"]

    if include_mi:
        acquisition_functions += [mi_ws_acquisition]
        refs += ["MI_WS"]

    if fit:
        for e in range(args.nb_experiments):
            data = data_loader(store_file=store_file)

            for ref, acq_fun in zip(refs, acquisition_functions):

                main_experiment(GPModel, rbf_kernel, kernel_params, data, prior_var, annotator.annotate,
                                acq_fun, cost_fun, precision_fun, args, var_factor=var_factor,
                                betas=1 - np.linspace(0.0, 1.0, 100), dir=args.save_dir + ref + suff, e=e,
                                fit_hyperparams=fit_hyp, standardise=True)

    vis_res_main(args, suff, interpolate_results=interpolate_results, include_mi=include_mi,
                 group="uci" + suff)


if __name__ == '__main__':

    suff = "_conc_a_test"

    uci_exp_main(suff, lambda a: a, lambda c: c)
