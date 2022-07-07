import tensorflow as tf
import numpy as np

from src.experiment_utils import rbf_kernel
from src.experiment_utils import constant_noise, transform_noise, scale_noise


class GPModel:

    def __init__(self, X_train, y_train, prior_mean=0.0, prior_var=1.0, var_factor=0.0, precision=None,
                 kernel=rbf_kernel, kernel_params=np.ones((2,)), noise_func=constant_noise, stat=None):

        if precision is None:
            precision = tf.zeros((X_train.shape[0], 1), dtype=tf.float64)

        assert X_train.shape[0] == y_train.shape[0] and X_train.shape[0] == precision.shape[0]

        # TODO: Remove this if not used
        self.prior_mean = tf.constant(prior_mean, dtype=tf.float64)

        self.X_train = tf.convert_to_tensor(X_train)

        # We train model to predict deviation from mean, but add the mean back in prediction
        self.y_train = tf.convert_to_tensor(y_train) - self.prior_mean
        self.precision = tf.convert_to_tensor(precision, dtype=tf.float64)

        self._kernel = kernel

        self.prior_var = tf.constant(prior_var, dtype=tf.float64)
        self.var_factor = tf.constant(var_factor, dtype=tf.float64)
        self.kernel_params = tf.constant(kernel_params, dtype=tf.float64)

        self.noise_func = noise_func
        self.scale_noise = scale_noise

        if stat is not None:
            self.stat = tf.convert_to_tensor(stat)
        else:
            self.stat = None

        self.inv_cov, self.cov = self._calculate_inverse_covariance(self.stat)

    def _calculate_inverse_covariance(self, stat=None):
        if stat is None:
            return self._calculate_inverse_covariance_spec(self.kernel_params, return_kernel=True)
        else:
            return self._calculate_inverse_covariance_from_stat_spec(self.kernel_params, stat, return_kernel=True)

    def _calculate_inverse_covariance_spec(self, kp, return_kernel=False):
        cov = self._kernel(tf.reshape(self.X_train, [self.X_train.shape[0], 1, self.X_train.shape[-1]]),
                           tf.reshape(self.X_train, [1, self.X_train.shape[0], self.X_train.shape[-1]]), kp)

        return self._calculate_inverse_covariance_from_cov(cov, return_kernel=return_kernel)

    def _calculate_inverse_covariance_from_stat_spec(self, kp, stat, return_kernel=False):
        cov = self._kernel(stat, kp)

        return self._calculate_inverse_covariance_from_cov(cov, return_kernel=return_kernel)

    def _calculate_inverse_covariance_from_cov(self, cov, return_kernel=False):

        inv_cov = tf.linalg.inv(
            cov + tf.linalg.diag(self.calculate_noise_covariance(self.precision, self.X_train)[:, 0]))

        if return_kernel:
            return inv_cov, cov
        else:
            return inv_cov

    def calculate_noise_covariance(self, precision, X):
        return transform_noise(self.noise_func(X, self.prior_var), self.var_factor, precision)

    def train(self, num_epochs, X_valid, y_valid, valid_precision, lr=1e-1, verbose=True, tol=5e-2, gtol=1e-5):
        # Update model parameters

        X_valid, y_valid, valid_precision = tf.convert_to_tensor(X_valid), tf.convert_to_tensor(y_valid), \
                                            tf.convert_to_tensor(valid_precision)

        curr_params = self.train_inner(num_epochs, X_valid, y_valid, valid_precision,
                                       loss_fun = self.calculate_loss, lr=lr, verbose=verbose, tol=tol, gtol=gtol)
        self.kernel_params = curr_params
        self.inv_cov, self.cov = self._calculate_inverse_covariance()

        train_loss, valid_loss = self.calculate_final_loss(X_valid, y_valid, valid_precision)

        return train_loss, valid_loss

    def train_from_stat(self, num_epochs, lr=1e-1, verbose=True, tol=5e-2, gtol=1e-5):
        # Update model parameters

        curr_params = self.train_inner(num_epochs, None, None, None, loss_fun=self.calculate_loss_from_stat,
                                       lr=lr, verbose=verbose, tol=tol, gtol=gtol)
        self.kernel_params = curr_params
        self.inv_cov, self.cov = self._calculate_inverse_covariance(stat=self.stat)

        train_loss = self.calculate_loss_from_stat(self.kernel_params)
        return train_loss, 0

    def train_inner(self, num_epochs, X_valid, y_valid, valid_precision, loss_fun, lr=1e-1, verbose=True, tol=5e-2,
                    gtol=1e-5):
        """ Learn hyperparameters of kernel """

        kp = tf.Variable(self.kernel_params)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        old_loss = np.inf
        curr_params = tf.convert_to_tensor(kp)
        for epoch in range(num_epochs):
            kp, optimizer, grads = self.train_epoch(kp, optimizer, loss_fun)

            train_loss = loss_fun(tf.convert_to_tensor(kp)).numpy()
            if verbose:
                print("Train loss at epoch {}: {}".format(epoch + 1, train_loss))

                if X_valid is not None:
                    valid_loss = self.calculate_loss_new_data(X_valid, y_valid, valid_precision,
                                                              tf.convert_to_tensor(kp)).numpy()
                    print("Validation loss at epoch {}: {}".format(epoch + 1, valid_loss))

            if (epoch > 0) and ((old_loss - train_loss) / np.max((np.abs(old_loss), np.abs(train_loss), 1)) <= tol
                                or tf.math.reduce_max(grads) < gtol):
                # Stop training if loss has not improved/has increased or if gradient is small
                curr_params = tf.convert_to_tensor(kp)
                break
            elif (epoch + 1) == num_epochs:
                print("Maximum number of epochs reached")
                curr_params = tf.convert_to_tensor(kp)

            old_loss = train_loss

        return curr_params

    def calculate_final_loss(self, X_valid, y_valid, valid_precision):
        train_loss = self.calculate_loss(self.kernel_params).numpy()
        valid_loss = self.calculate_loss_new_data(X_valid, y_valid, valid_precision, self.kernel_params).numpy()

        return train_loss, valid_loss

    def train_epoch(self, kp, optimizer, loss_fun):
        """ Learn hyperparameters of kernel """

        with tf.GradientTape() as tape:
            # Calculate loss
            loss = loss_fun(kp)

        # Calculate gradients
        grads = tape.gradient(loss, [kp])

        # Update parameters
        opt = optimizer.apply_gradients(zip(grads, [kp]))

        return kp, optimizer, grads

    def neg_log_likelihood(self, y, inv_cov=None, diag_cov=False):

        if diag_cov:
            log_likelihood_norm = - 0.5 * tf.reduce_sum(tf.linalg.tensor_diag_part(inv_cov))
        else:
            log_likelihood_norm = - 0.5 * tf.linalg.logdet(inv_cov)  #tf.math.log(tf.linalg.det(inv_cov))  # + constant

        log_likelihood = 0.5 * tf.matmul(tf.matmul(tf.transpose(y), inv_cov), y)

        return log_likelihood_norm + log_likelihood

    def calculate_loss_wrapper(self, kp):
        return self.calculate_loss(tf.convert_to_tensor(kp)).numpy()[0]

    def calculate_loss(self, kp):
        inv_cov = self._calculate_inverse_covariance_spec(kp)

        return self.neg_log_likelihood(self.y_train, inv_cov)

    def calculate_loss_from_stat(self, kp):
        inv_cov = self._calculate_inverse_covariance_from_stat_spec(kp, self.stat)

        return self.neg_log_likelihood(self.y_train, inv_cov)

    def calculate_loss_new_data(self, X_new, y_new, precision_new, kp):

        inv_cov = self._calculate_inverse_covariance_spec(kp)

        k = self._kernel(tf.reshape(self.X_train, [self.X_train.shape[0], 1, self.X_train.shape[-1]]),
                         tf.reshape(X_new, [1, X_new.shape[0], X_new.shape[-1]]), self.kernel_params)

        c = self._kernel(X_new, X_new, self.kernel_params) + self.calculate_noise_covariance(precision_new, X_new)[:, 0]

        part = tf.linalg.matmul(tf.transpose(k), inv_cov)
        m = tf.linalg.matmul(part, self.y_train)

        # Assume independence
        s = tf.expand_dims(c - tf.linalg.tensor_diag_part(tf.linalg.matmul(part, k)), axis=-1)
        inv_cov_new = tf.linalg.diag(1 / s[:, 0])  #tf.linalg.inv(tf.linalg.diag(s[:, 0]))

        return self.neg_log_likelihood(y_new - m, inv_cov_new, diag_cov=True)

    def single_predict(self, X_new, k=None):
        """ Predict on new data """
        X_new = tf.convert_to_tensor(X_new)

        if k is None:
            k = tf.expand_dims(self._kernel(self.X_train, X_new, self.kernel_params), axis=-1)

        c = self._kernel(X_new, X_new, self.kernel_params) \
            + self.calculate_noise_covariance(np.array([[0.0]]), X_new)[:, 0]

        m = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(k), self.inv_cov), self.y_train)
        s = c - tf.linalg.matmul(tf.linalg.matmul(tf.transpose(k), self.inv_cov), k)

        return (m + self.prior_mean).numpy(), s.numpy()

    def predict(self, X_new, return_std=True):
        # Predict with precision zero
        return self.noisy_predict(X_new, tf.zeros((X_new.shape[0], 1), dtype=tf.float64), return_std)

    def predict_from_stat(self, X_new, k_stat, c_stat, return_std=True):
        # Predict with precision zero
        return self.noisy_predict_from_stat(X_new, tf.zeros((X_new.shape[0], 1), dtype=tf.float64), k_stat, c_stat, return_std)

    def noisy_predict(self, X_new, precision, return_std=True):
        X_new = tf.convert_to_tensor(X_new)

        k = self._kernel(tf.reshape(self.X_train, [self.X_train.shape[0], 1, self.X_train.shape[-1]]),
                         tf.reshape(X_new, [1, X_new.shape[0], X_new.shape[-1]]), self.kernel_params)

        c = self._kernel(X_new, X_new, self.kernel_params)

        return self.noisy_predict_inner(X_new, precision, k, c, return_std)

    def noisy_predict_from_stat(self, X_new, precision, k_stat, c_stat, return_std=True):
        X_new = tf.convert_to_tensor(X_new)
        k, c = self.kernel_from_stat(tf.convert_to_tensor(k_stat)), self.kernel_from_stat(tf.convert_to_tensor(c_stat))

        return self.noisy_predict_inner(X_new, precision, k, c, return_std)
    
    def noisy_predict_inner(self, X_new, precision, k, c, return_std):
        """ Predict with lower precision """

        part = tf.linalg.matmul(tf.transpose(k), self.inv_cov)
        m = tf.linalg.matmul(part, self.y_train)

        if return_std:

            c += self.calculate_noise_covariance(precision, X_new)[:, 0]
            s = tf.expand_dims(c - tf.reduce_sum(part * tf.transpose(k), axis=-1), axis=-1)

            return (m + self.prior_mean).numpy(), s.numpy()
        else:
            return (m + self.prior_mean).numpy()

    def predict_model_var(self, X_new):
        X_new = tf.convert_to_tensor(X_new)

        k = self._kernel(tf.reshape(self.X_train, [self.X_train.shape[0], 1, self.X_train.shape[-1]]),
                         tf.reshape(X_new, [1, X_new.shape[0], X_new.shape[-1]]), self.kernel_params)

        c = self._kernel(X_new, X_new, self.kernel_params)

        return self.predict_model_var_inner(k, c)

    def predict_model_var_from_stat(self, k_stat, c_stat):

        k, c = self.kernel_from_stat(tf.convert_to_tensor(k_stat)), self.kernel_from_stat(tf.convert_to_tensor(c_stat))

        return self.predict_model_var_inner(k, c)

    def predict_model_var_inner(self, k, c):
        """ Predict with lower precision, return only model variance """
        s = tf.expand_dims(c - tf.reduce_sum(k * tf.linalg.matmul(self.inv_cov, k), axis=0), axis=-1)

        return s.numpy()

    def var_noisy_y_given_f(self, X, precision=0.0):
        return self.calculate_noise_covariance(precision, X).numpy()

    def var_y_given_f(self, X):
        return self.var_noisy_y_given_f(X)

    def var_noisy_y_given_y(self, var_y, cov_y, precision=0.0):
        # var_y: variance of y given x
        # cov_y: covariance of ~y and y
        return (var_y + self.scale_noise(self.var_factor, precision) - cov_y**2 / var_y).numpy()

    def var_noisy_y_given_y_ext(self, X, var_y, cov_y, precision=0.0):
        # var_y: variance of y given x
        # cov_y: covariance of ~y and y
        return self.var_noisy_y_given_y(var_y, cov_y, precision)

    def kernel(self, x_m, x_n):
        return self._kernel(x_m, x_n, self.kernel_params)

    def kernel_from_stat(self, x_stat):
        return self._kernel(x_stat, self.kernel_params)


class GPModelII(GPModel):

    def __init__(self, X_train, y_train, prior_mean=0.0, prior_var=1.0, var_factor=1.0, precision=None, kernel=rbf_kernel,
                 kernel_params=np.ones((2,)), noise_func=constant_noise, stat=None):

        super().__init__(X_train, y_train, prior_mean, prior_var, var_factor, precision, kernel, kernel_params,
                         noise_func, stat=stat)

    def var_noisy_y_given_y(self, var_y, cov_y, precision=0.0):
        return self.scale_noise(self.var_factor, precision).numpy()




