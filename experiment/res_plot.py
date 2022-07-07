# http//www.scipy-lectures.org/intro/matplotlib/matplotlib.html
from pylab import *
from matplotlib import rc, rcParams
import tikzplotlib

from src.experiment_utils import parse_args


def vis_res_main(args, suff="", metric_name="MSE", interpolate_results=False, include_mi=False,
                 show_extra=False, group="sine_reg", log_scale=True):

    dir = args.save_dir

    nb_experiments = args.nb_experiments

    cost_general = np.cumsum(np.load(dir + "Bald" + suff + "_costs_experiment_" + str(0) + ".npy"))
    costs_bald_ws = []

    Bald_mean, eval_Bald_mean, Bald_WS_mean, eval_Bald_WS_mean, Random_mean = [], [], [], [], []

    if include_mi:
        MI_WS_mean = []

    if interpolate_results:
        costs_bald_ws_saved = []
        costs_mi_ws_saved = []

    for i in np.arange(0, nb_experiments):
        Bald = np.load(dir + "Bald" + suff + "_" + metric_name + "_results_experiment_" + str(i) + ".npy")
        Bald_WS = np.load(dir + "Bald_WS" + suff + "_" + metric_name + "_results_experiment_" + str(i) + ".npy")
        costs_bald_ws = np.cumsum(np.load(dir + "Bald_WS" + suff + "_costs_experiment_" + str(i) + ".npy"))
        Bald_WS_precision = np.load(dir + "Bald_WS" + suff + "_precision_experiment_" + str(i) + ".npy")
        Random = np.load(dir + "Random" + suff + "_" + metric_name + "_results_experiment_" + str(i) + ".npy")

        print("Min precision parameter MI with f {}".format(Bald_WS_precision.min()))
        print("Max precision parameter MI with f {}".format(Bald_WS_precision.max()))
        print("Proportion of lowest precision parameter MI with f {}".format((Bald_WS_precision == 0.0).mean()))
        print("Proportion of highest precision parameter MI with f {}".format((Bald_WS_precision == 1.0).mean()))

        Bald_mean.append(Bald)
        Random_mean.append(Random)

        if include_mi:
            MI_WS = np.load(dir + "MI_WS" + suff + "_" + metric_name + "_results_experiment_" + str(i) + ".npy")
            MI_WS_precision = np.load(dir + "MI_WS" + suff + "_precision_experiment_" + str(i) + ".npy")
            costs_mi_ws = np.cumsum(np.load(dir + "MI_WS" + suff + "_costs_experiment_" + str(i) + ".npy"))

            print("Min precision parameter MI with Y {}".format(MI_WS_precision.min()))
            print("Max precision parameter MI with Y {}".format(MI_WS_precision.max()))
            print("Proportion of lowest parameter precision MI with Y {}".format((MI_WS_precision == 0.0).mean()))
            print("Proportion of highest parameter precision MI with Y {}".format((MI_WS_precision == 1.0).mean()))

        plt.plot(cost_general, Bald, color="red", linewidth=3.0, marker="x", label=r"BALD")
        plt.plot(costs_bald_ws, Bald_WS, color="orange", linewidth=3.0, marker="x", label=r"MI($\tilde{Y};f$)")

        if include_mi:
            plt.plot(costs_mi_ws, MI_WS, color="green", linewidth=3.0, marker="x", label=r"MI($\tilde{Y};Y$)")

        if interpolate_results:
            costs_bald_ws_saved = costs_bald_ws.copy()
            costs_bald_ws = np.linspace(cost_general[0], cost_general[-1], len(cost_general) * 10)
            Bald_WS = np.interp(costs_bald_ws, costs_bald_ws_saved, Bald_WS)
            Bald_WS_mean.append(Bald_WS)
            plt.plot(costs_bald_ws, Bald_WS, color="brown", linewidth=3.0, marker=None, label=r"MI($\tilde{Y};f$) alt")

            if include_mi:
                costs_mi_ws_saved = costs_mi_ws.copy()
                costs_mi_ws = np.linspace(cost_general[0], cost_general[-1], len(cost_general) * 10)
                MI_WS = np.interp(costs_mi_ws, costs_mi_ws_saved, MI_WS)
                MI_WS_mean.append(MI_WS)
                plt.plot(costs_mi_ws, MI_WS, color="brown", linewidth=3.0, marker=None, label=r"MI($\tilde{Y};Y$) alt")

        else:
            Bald_WS_mean.append(Bald_WS)

            if include_mi:
                MI_WS_mean.append(MI_WS)


        plt.plot(cost_general, Random, color="blue", linewidth=3.0, marker="x", label=r"\textbf{Random}")

        plt.xlabel(r"Cost")
        plt.ylabel(r"{}".format(metric_name))
        plt.grid()
        plt.legend(loc=4)

        if show_extra:
            plt.show()
        else:
            plt.close()


        start_pool = int(cost_general[0])
        plt.scatter(np.linspace(1, Bald_WS_precision.shape[0] - start_pool, Bald_WS_precision.shape[0] - start_pool),
                    Bald_WS_precision[start_pool:], color="orange", linewidth=3.0, marker="x", label=r"MI($\tilde{Y};f$)")

        if include_mi:
            plt.scatter(np.linspace(1, MI_WS_precision.shape[0] - start_pool, MI_WS_precision.shape[0] - start_pool),
                        MI_WS_precision[start_pool:], color="green", linewidth=3.0, marker="x", label=r"MI($\tilde{Y};Y$)")

        plt.xlabel(r"Iteration")
        plt.ylabel(r"Annotation precision")
        plt.legend()

        #if i in [0, 1]:
        #   tikzplotlib.save(dir + "BALD_WS_prec_ex_" + str(i) + ".tex")

        if show_extra:
            plt.show()
        else:
            plt.close()


    # activate latex text rendering
    rc("text", usetex=True)
    rc("axes", linewidth=2)
    rc("font", weight="bold")
    # rcParams["text.latex.preamble"] = [r"\usepackage{sfmath} \boldmath"]

    print(cost_general.shape)
    Bald_mean, Random_mean = np.column_stack(Bald_mean), np.column_stack(Random_mean)
    plot_final(cost_general, Bald_mean, "red", "x", r"BALD")

    if interpolate_results:
        ws_marker = None
    else:
        ws_marker = "x"

    Bald_WS_mean = np.column_stack(Bald_WS_mean)
    plot_final(costs_bald_ws, Bald_WS_mean, "orange", ws_marker, r"MI($\tilde{Y};f)$")

    if include_mi:
        MI_WS_mean = np.column_stack(MI_WS_mean)
        plot_final(costs_mi_ws, MI_WS_mean, "green", ws_marker, r"MI$(\tilde{Y};Y)$")

    plot_final(cost_general, Random_mean, "blue", "x", r"Random")

    plt.xlabel(r"Cost")
    plt.ylabel(r"{}".format(metric_name))
    plt.grid()
    plt.legend(loc=4) #1

    if log_scale:
        plt.yscale("log")

    tikzplotlib.save(dir + group + "_exp" + suff + ".tex")

    plt.show()


def plot_final(x, y, col, marker, label):
    plt.plot(x, np.median(y, axis=-1), color=col, linewidth=3.0, marker=marker, label=label)
    plt.fill_between(x, np.quantile(y, 0.25, axis=-1), np.quantile(y, 0.75, axis=-1),
                     alpha=0.1, color=col)



if __name__ == "__main__":
    args = parse_args()
    vis_res_main(args)