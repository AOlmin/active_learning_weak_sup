from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import random

# from src.experiment_utils import parse_args, rbf_kernel, \
# 	get_sine_regression_data, SineRegressionAnnotator
# from src.gp_model import GPModel
#
# from src.experiment_utils import parse_args, rbf_kernel, get_simple_class_data, SimpleClassAnnotator, acc_metric
# from src.gpc_model import GPClassifier
#
# from src.experiment_base_gp import main_experiment


def random_acquisition(X_Pool, Queries, **kwargs):

	x_pool_index = np.asarray(random.sample(range(0, X_Pool.shape[0]), Queries))

	return x_pool_index, (np.ones((x_pool_index.shape[0])) * (-1)).astype(int)



# if __name__ == '__main__':
#
# 	regression = False
# 	if regression:
# 		# Data params
# 		rel_amp, fs, var = 0.2, 3, 0.01  # fs=7
# 		prior_var = 0.01
#
# 		store_file = "reg_data_rel_amp_" + str(rel_amp) + "_fs_" + str(fs) + "_var_" + str(var) + "_test_exp_" + str(0)
#
# 		data = get_sine_regression_data(store_file=store_file, rel_amp=rel_amp, fs=fs, reuse_data=False,
# 										num_samples=10000, non_uniform=True)
#
# 		annotator = SineRegressionAnnotator(rel_amp, fs, var)
#
# 		args = parse_args()
# 		kernel_params = np.ones((2,))
# 		main_experiment(GPModel, rbf_kernel, kernel_params, data, prior_var, annotator.annotate,
# 						random_acquisition, lambda a: a, lambda c: c, args,
# 						dir=args.save_dir + "random_test", e=0, ref_model=False)
#
# 	else:
#
# 		mean_1, mean_2 = -0.5 * np.ones((2,)), 0.5 * np.ones((2,))
# 		store_file = "data/class_data_mean_1_" + str(mean_1[0]) + "_" + str(mean_1[1]) + "_mean_2_" + str(mean_2[0]) \
# 					 + "_" + str(mean_2[1]) + "_test"
#
# 		data = get_simple_class_data(store_file=store_file, mean_1=mean_1, mean_2=mean_2, reuse_data=True,
# 									 num_samples=10000)
#
# 		annotator = SimpleClassAnnotator(mean_1, mean_2)
#
# 		args = parse_args()
# 		kernel_params = np.ones((2,))
# 		main_experiment(GPClassifier, rbf_kernel, kernel_params, data, 0.0, annotator.annotate,
# 						random_acquisition, lambda a: a, lambda c: c, args, metric=acc_metric, metric_name='ACC',
# 						dir=args.save_dir + "GPC_Random_test", e=0, ref_model=False)










