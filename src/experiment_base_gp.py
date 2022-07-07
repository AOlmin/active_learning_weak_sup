import numpy as np
from src.experiment_utils import mse_metric, standardise_data, constant_noise, identity_function
from src.acq_fun_utils import can_handle_noisy_annotations


def main_experiment(model_class, kernel, kernel_params, data, prior_var, labelling_function, acquisition_function,
                    label_cost, cost_precision, args, prior_mean=0.0, var_factor=1.0, betas=np.linspace(0.1, 1, 100),
                    metric=mse_metric, metric_name='MSE', dir="res_test/", e=0, fit_hyperparams=False,
                    noise_func=constant_noise, preprocess_function=identity_function,
                    standardise=False, init_precision=None):

    nb_epoch = args.nb_epochs

    acquisition_iterations = args.nb_acquisitions

    queries = args.nb_queries

    max_b = queries * acquisition_iterations
    b_final = 0

    print('Starting experiment number ', e)

    X_train, y_train, X_valid, y_valid, X_test, y_test, X_Pool, y_Pool = data

    if standardise:
        mean_st, std_st = X_train.mean(axis=0), X_train.std(axis=0)
        prior_mean = y_train.mean()
    else:
        mean_st, std_st = 0, 1

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    x_pool_All = np.zeros(shape=(1))


    if init_precision is None:
        precision = np.ones((X_train.shape[0], 1)) * betas[-1]
    else:
        precision = init_precision

    model = model_class(standardise_data(X_train, mean_st, std_st),
                        y_train, prior_mean=prior_mean, prior_var=prior_var, var_factor=var_factor,
                        precision=precision, kernel=kernel, kernel_params=kernel_params, noise_func=noise_func)

    if fit_hyperparams:
        train_loss, valid_loss = model.train(nb_epoch, standardise_data(X_valid, mean_st, std_st),
                                                       y_valid, np.ones((X_valid.shape[0], 1)) * betas[-1])
    else:
        train_loss, valid_loss = model.calculate_final_loss(standardise_data(X_valid, mean_st, std_st), y_valid,
                                                                      np.ones((X_valid.shape[0], 1)) * betas[-1])

    preds = model.predict(standardise_data(X_test, mean_st, std_st))
    score = metric(preds, y_test)
    print('Test {}: {}'.format(metric_name, score))

    all_score = score

    costs_all = np.sum(label_cost(precision))[np.newaxis]
    b = 0

    print('Starting Active Learning in Experiment ', e)
    i = 0
    while b < (acquisition_iterations * queries):

        print('POOLING ITERATION', i)

        if (max_b - b) >= min(label_cost(betas)) and X_Pool.size:  # Only try to label more instances if we can afford it and if there are more to label

            x_pool_index, beta_pool_index = acquisition_function(standardise_data(X_Pool, mean_st,
                                                                                   std_st), queries,
                                                                 model=model,
                                                                 betas=betas,
                                                                 label_cost=label_cost)

            # If we can't afford to label all queried points; remove points until we can, or select lower precision
            while (b + np.sum(label_cost(betas[beta_pool_index]))) > max_b:
                if np.isscalar(x_pool_index) or x_pool_index.shape[0] == 1:
                    # Can't afford to label even one data point
                    # See if we can afford a cheaper option

                    diff = max_b - b
                    beta_option = cost_precision(diff)

                    if can_handle_noisy_annotations and min(betas) <= beta_option <= max(betas):
                            # Annotate instances one by one at this point (even if it might turn out to be better to
                            # e.g. annotate several instances with even lower precision)
                            available_betas = label_cost(betas) <= label_cost(beta_option)
                            x_pool_index, beta_pool_index = acquisition_function(standardise_data(X_Pool, mean_st,
                                                                                                  std_st), 1,
                                                                                 model=model,
                                                                                 betas=betas[available_betas],
                                                                                 label_cost=label_cost)
                            beta_pool_index = (np.arange(0, betas.shape[0], 1)[available_betas])[beta_pool_index]

                    else:
                        # We cannot afford to label any more instances (this we should catch before even doing acquisitions)
                        b_final = b
                        b = max_b
                        x_pool_index, beta_pool_index = [], []
                        break
                else:
                    x_pool_index = x_pool_index[1:]
                    beta_pool_index = beta_pool_index[1:]


            # Only label more data points if we can afford it (perhaps redundant)
            if (b + np.sum(label_cost(betas[beta_pool_index]))) <= max_b:
                label_precision = betas[beta_pool_index]
                b += np.sum(label_cost(label_precision))

                x_pool_All = np.append(x_pool_All, x_pool_index)

                Pooled_X = X_Pool[x_pool_index, :]
                Pooled_Y = y_Pool[x_pool_index]

                precision = np.concatenate((precision, label_precision[:, np.newaxis]), axis=0)

                X_Pool = np.delete(X_Pool, x_pool_index, axis=0)
                y_Pool = np.delete(y_Pool, x_pool_index, axis=0)

                print('Acquised points added to training set, used budget: {}'.format(b))

                Pooled_X = preprocess_function(Pooled_X)

                # Here we assume that labels are generated prior to standardisation (if we standardise)
                new_labels = labelling_function(Pooled_X, Pooled_Y, label_precision[:, np.newaxis])

                X_train = np.concatenate((X_train, Pooled_X), axis=0)
                y_train = np.concatenate((y_train, new_labels), axis=0)

                # For diagnostics
                np.save(dir + '_X_train_' + str(e), X_train)
                np.save(dir + '_Y_train_' + str(e), y_train)

                if standardise:
                    mean_st, std_st = X_train.mean(axis=0), X_train.std(axis=0)
                    prior_mean = y_train.mean()

                model = model_class(standardise_data(X_train, mean_st, std_st), y_train, prior_mean=prior_mean,
                                    prior_var=prior_var, var_factor=var_factor, precision=precision, kernel=kernel,
                                    kernel_params=kernel_params, noise_func=noise_func)

                if fit_hyperparams:
                    Train_Loss, Valid_Loss = model.train(nb_epoch, standardise_data(X_valid, mean_st,
                                                                                    std_st), y_valid,
                                                         np.ones((X_valid.shape[0], 1)) * betas[-1])
                else:
                    Train_Loss, Valid_Loss = model.calculate_final_loss(standardise_data(X_valid, mean_st,
                                                                                    std_st), y_valid,
                                                                        np.ones((X_valid.shape[0], 1)) * betas[-1])

                valid_loss = np.append(valid_loss, Valid_Loss, axis=1)
                train_loss = np.append(train_loss, Train_Loss, axis=1)
                costs_all = np.concatenate((costs_all, np.sum(label_cost(label_precision))[np.newaxis]))

                print('Evaluate Model Test Metric with pooled points')

                preds = model.predict(standardise_data(X_test, mean_st, std_st))
                score = metric(preds, y_test)
                print('Test {}: {}'.format(metric_name, score))
                all_score = np.append(all_score, score)

                if i % 10 == 0:  # Just a safeguard
                    print("Saving results at iteration: {}".format(i))
                    np.save(dir + '_train_loss_' + 'experiment_' + str(e) + '.npy', train_loss)
                    np.save(dir + '_valid_loss_' + 'experiment_' + str(e) + '.npy', valid_loss)
                    np.save(dir + '_' + metric_name + '_results_' + 'experiment_' + str(e) + '.npy', all_score)
                    np.save(dir + '_precision_' + 'experiment_' + str(e) + '.npy', precision)
                    np.save(dir + '_costs_' + 'experiment_' + str(e) + '.npy', costs_all)

                print('Use this trained model with pooled points for training again')

                i += 1

            b_final = b
        else:
            if not X_Pool.size:
                print("Data pool exhausted")

            b_final = b
            b = max_b  # break

    print("Used budget out of {}: {}".format(max_b, b_final))

    print('Saving results')
    np.save(dir + '_train_loss_' + 'experiment_' + str(e) + '.npy', train_loss)
    np.save(dir + '_valid_loss_' + 'experiment_' + str(e) + '.npy', valid_loss)
    np.save(dir + '_' + metric_name + '_results_' + 'experiment_' + str(e) + '.npy', all_score)
    np.save(dir + '_precision_' + 'experiment_' + str(e) + '.npy', precision)
    np.save(dir + '_costs_' + 'experiment_' + str(e) + '.npy', costs_all)








