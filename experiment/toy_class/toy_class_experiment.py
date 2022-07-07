from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from src.experiment_utils import parse_args
from experiment.toy_class.toy_class_experiment_base import toy_class_exp_main


if __name__ == '__main__':
    args = parse_args()
    version = args.version
    q = args.q

    suff = "_toy_" + version + "a_" + str(q)

    base_cost = 0.1
    cost_factor = 0.9
    toy_class_exp_main(args, suff, lambda a: base_cost + cost_factor * a ** q,
                       lambda c: ((c - base_cost) / cost_factor) ** (1 / q), fit=args.fit, include_mi=True, version=version,
                       interpolate_results=args.interpolate)

