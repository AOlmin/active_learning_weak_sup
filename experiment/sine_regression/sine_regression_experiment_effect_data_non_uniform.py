from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from experiment.sine_regression.sine_reg_experiment_base import sine_reg_exp_main


from src.experiment_utils import parse_args

if __name__ == '__main__':

    args = parse_args()
    fs = args.fs

    rel_amp = 0.2
    prior_var = 0.01
    var_factor = prior_var * 9
    suff = "_fs_" + str(fs).replace('.0', '') + "_a_nonun"

    q = 1
    sine_reg_exp_main(args, rel_amp, fs, prior_var, var_factor, suff, lambda a: 1 / (1 + ((var_factor / prior_var) * a))**q,
                      lambda c: prior_var * ((1/c)**(1 / q) - 1) / var_factor, fit=args.fit, non_uniform=True,
                      include_mi=True, interpolate_results=args.interpolate)

