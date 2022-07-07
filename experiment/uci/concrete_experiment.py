from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from experiment.uci.uci_experiment_base import uci_exp_main
from experiment.uci.utils.concrete_data import get_concrete_data, ConcreteAnnotator

from src.experiment_utils import parse_args


if __name__ == '__main__':
    args = parse_args()

    # experiment/uci/data
    store_file = args.data_dir + "concrete.xlsx"

    suff = "_conc"

    if args.fit_hyp:
        suff += "_fh"

    prior_var = 1e-3
    cost_factor = 9
    var_factor = 1

    q = args.q
    uci_exp_main(args, store_file, get_concrete_data, ConcreteAnnotator, prior_var, suff,
                 lambda a: 1 / (1 + cost_factor * a)**q,
                 lambda c: ((1/c)**(1 / q) - 1) / cost_factor, var_factor=var_factor, fit=args.fit, fit_hyp=args.fit_hyp,
                 interpolate_results=args.interpolate)
