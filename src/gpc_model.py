import tensorflow as tf
import numpy as np

from src.experiment_utils import rbf_kernel, transform_noise, constant_noise
from scipy.stats import norm as scipy_norm


class GPClassifier:

    def __init__(self, X_train, y_train, prior_mean=0.0, prior_var=0.0, var_factor=1.0, precision=None,
                 kernel=rbf_kernel, kernel_params=np.ones((2,)), noise_func=constant_noise):

        if precision is None:
            precision = tf.ones((X_train.shape[0], 1), dtype=tf.float64)

        assert X_train.shape[0] == y_train.shape[0] and X_train.shape[0] == precision.shape[0]

        self.X_train = tf.convert_to_tensor(X_train)
        self.y_train = tf.convert_to_tensor(y_train)
        self._kernel = kernel

        self.prior_mean = prior_mean  # TODO: Not used
        self.prior_var = tf.constant(prior_var, dtype=tf.float64)
        self.var_factor = tf.constant(var_factor, dtype=tf.float64)
        self.kernel_params = tf.constant(kernel_params, dtype=tf.float64)

        self.noise_func = noise_func
        self.scaled_precision = self.scale_precision(self.X_train, tf.convert_to_tensor(precision, dtype=tf.float64))
        self.cov = self._calculate_covariance()
        self.nu, self.tau, self.L, self.z, self.log_marg = self._fit_model()

    def _calculate_covariance(self):
        return self._calculate_covariance_spec(self.kernel_params)

    def _calculate_covariance_spec(self, kp):
        cov = self._kernel(tf.reshape(self.X_train, [self.X_train.shape[0], 1, self.X_train.shape[-1]]),
                           tf.reshape(self.X_train, [1, self.X_train.shape[0], self.X_train.shape[-1]]), kp)

        return cov

    def _fit_model(self, max_it=500, tol=1e-2, nu=None, tau=None):
        """Get approximate posterior using EP (Alg. 3.5 in Rasmussen and Williams (2006))"""
        n = self.X_train.shape[0]

        if nu is None:
            nu = tf.zeros((n,), dtype=tf.float64)
        if tau is None:
            tau = tf.zeros((n,), dtype=tf.float64)

        nu, tau = tf.Variable(nu), tf.Variable(tau)

        mu = tf.zeros((n,), dtype=tf.float64)
        cov = tf.identity(self.cov)
        S_root, B0, L = None, None, None

        converged = False
        it = 0
        while not converged:
            nu_old, tau_old = tf.identity(tf.convert_to_tensor(nu)), tf.identity(tf.convert_to_tensor(tau))

            for i in range(n):

                # Compute cavity parameters
                tau_cavity = 1 / cov[i, i] - tau[i]
                nu_cavity = (1 / cov[i, i]) * mu[i] - nu[i]

                # Compute marginal moments
                mu_hat, sigma_hat = self._marginal_moments(nu_cavity, tau_cavity, i)

                # Update site parameters
                delta_tau = (1 / sigma_hat) - tau_cavity - tau[i]

                # Make sure variance is positive
                if (tau[i] + delta_tau) > 0:
                    tau[i].assign(tau[i] + delta_tau)
                    nu[i].assign((1 / sigma_hat) * mu_hat - nu_cavity)

                    cov = cov - (1 / ((1 / delta_tau) + cov[i, i])) * tf.linalg.tensordot(cov[:, i], cov[i, :], axes=0)
                    mu = tf.linalg.matvec(cov, nu)

            S_root, B0, L = self._update_extra_params_part_1(tau)

            try:
                V = tf.linalg.solve(L, tf.linalg.matmul(S_root, self.cov))
            except:
                print("Non-invertible matrix")

            #assert np.allclose(np.dot(np.matmul(S_root, self.cov), V), np.transpose(L))

            # This is done to avoid loss in numerical precision
            cov = self.cov - tf.linalg.matmul(tf.transpose(V), V)
            mu = tf.linalg.matvec(cov, nu)

            it += 1
            if it == max_it:
                print("Maximum number of iterations reached.")
                converged = True
            elif tf.math.reduce_all(nu - nu_old < tol) and tf.math.reduce_all(tau - tau_old < tol):
                print("EP converged at iteration {}.".format(it))
                converged = True


        # Calculate log_marginal
        nu, tau = tf.expand_dims(tf.convert_to_tensor(nu), axis=-1), tf.expand_dims(tf.convert_to_tensor(tau), axis=-1)
        log_marg = self.log_marg_likelihood(self.X_train, self.y_train, self.scaled_precision, self.cov, nu, tau)

        # Calculate additional matrix useful for prediction
        z = self._update_extra_params_part_2(S_root, B0, L, nu)

        return nu, tau, L, z, log_marg

    def _update_extra_params_part_1(self, tau):
        S_root = tf.linalg.diag(tau) ** (1 / 2)
        B0 = tf.linalg.matmul(S_root, self.cov)
        B = tf.eye(self.X_train.shape[0], dtype=tf.float64) + tf.linalg.matmul(B0, S_root)
        L = tf.linalg.cholesky(B)

        return S_root, B0, L

    def _update_extra_params_part_2(self, S_root, B0, L, nu):
        z0 = tf.linalg.solve(L, tf.linalg.matmul(B0, nu))
        z = tf.linalg.matmul(S_root, tf.linalg.solve(tf.transpose(L), z0))

        return z

    def _marginal_moments(self, nu_cavity, tau_cavity, ind):
        # Compute marginal moments
        mu_cavity = nu_cavity / tau_cavity
        sigma_cavity = 1 / tau_cavity

        z = self.y_train[ind, 0] * mu_cavity / np.sqrt(1 + sigma_cavity)
        z_hat = (2 * self.scaled_precision[ind, 0] - 1) * scipy_norm.cdf(z) + (1 - self.scaled_precision[ind, 0])
        norm_z = scipy_norm.pdf(z)

        mu_hat = mu_cavity + (2 * self.scaled_precision[ind, 0] - 1) * self.y_train[ind, 0] * sigma_cavity * norm_z \
                 / (z_hat * np.sqrt(1 + sigma_cavity))
        sigma_hat = sigma_cavity - ((2 * self.scaled_precision[ind, 0] - 1) * (sigma_cavity ** 2) * norm_z
                                    / ((1 + sigma_cavity) * z_hat)) \
                    * (z + ((2 * self.scaled_precision[ind, 0] - 1) * norm_z / z_hat))

        return mu_hat, sigma_hat

    def scale_precision(self, X, precision):
        return transform_noise(self.noise_func(X, self.prior_var), self.var_factor, precision)

    def train(self, num_epochs, X_valid, y_valid, valid_precision, lr=1e-1, verbose=True, verbose_hp=True, hp_tol=5e-2,
              hp_gtol=1e-5, fm_tol=1e-2, tol=5e-2, fm_max_it=500, max_it=50):

        # We use the marginal log-likelihood as indicator of convergence
        old_loss, train_loss = np.inf, 0
        it = 0
        while it < max_it:
            # Fit Gaussian approximation
            self.nu, self.tau, self.L, self.z, self.log_marg = self._fit_model(max_it=fm_max_it, tol=fm_tol)

            # Fit hyperparameters
            train_loss, valid_loss = self.train_hp(num_epochs, X_valid, y_valid, valid_precision,
                                                   lr=lr, verbose=verbose_hp, tol=hp_tol, gtol=hp_gtol)

            it += 1
            if verbose:
                print("Train loss at iteration {}: {}.".format(it, train_loss))

            if (old_loss - train_loss) / np.max((np.abs(old_loss), np.abs(train_loss), 1)) <= tol:
                print("Training converged after {} iterations.".format(it))
                break

            elif it == max_it:
                print("Maximum iterations reached")

            old_loss = train_loss

        self.log_marg = - train_loss # self.log_marg_likelihood(self.y_train, self.precision, self.cov, self.nu, self.tau)

        S_root, B0, self.L = self._update_extra_params_part_1(self.tau[:, 0])
        self.z = self._update_extra_params_part_2(S_root, B0, self.L, self.nu)

        return train_loss, np.zeros((1, 1))

    def train_hp(self, num_epochs, X_valid, y_valid, valid_precision, lr=1e-1, verbose=False, tol=5e-2, gtol=1e-5):
        """ Learn hyperparameters of kernel """

        X_valid, y_valid, valid_precision = tf.convert_to_tensor(X_valid), tf.convert_to_tensor(y_valid), \
                                            tf.convert_to_tensor(valid_precision)

        kp = tf.Variable(self.kernel_params)

        train_loss, valid_loss = 0, 0
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        old_loss = np.inf
        curr_params = tf.convert_to_tensor(kp)
        for epoch in range(num_epochs):
            kp, optimizer, grads = self.train_epoch(kp, optimizer)


            train_loss = self.calculate_loss(tf.convert_to_tensor(kp))

            if verbose:
                print("Train loss at epoch {}: {}".format(epoch + 1, train_loss))
                #print("Validation loss at epoch {}: {}".format(epoch + 1, valid_loss))

            if (epoch > 0) and ((old_loss - train_loss) / np.max((np.abs(old_loss), np.abs(train_loss), 1)) <= tol
                                 or tf.math.reduce_max(grads) < gtol):
                 # Stop training if loss has not improved/has increased or if gradient is small
                 print("Hyperparameter training converged at epoch {}.".format(epoch + 1))
                 curr_params = tf.convert_to_tensor(kp)
                 break
            elif (epoch + 1) == num_epochs:
                print("Maximum number of epochs reached")
                curr_params = tf.convert_to_tensor(kp)

            old_loss = train_loss

        # Update model parameters
        self.kernel_params = curr_params
        self.cov = self._calculate_covariance()

        if not verbose:
            train_loss, valid_loss = self.calculate_final_loss(X_valid, y_valid, valid_precision)

        return train_loss, valid_loss

    def train_epoch(self, kp, optimizer):
        """ Learn hyperparameters of kernel """

        with tf.GradientTape() as tape:
            # Calculate loss
            loss = self.calculate_loss(kp)

        # Calculate gradients
        grads = tape.gradient(loss, [kp])

        # Update parameters
        opt = optimizer.apply_gradients(zip(grads, [kp]))

        return kp, optimizer, grads

    def calculate_final_loss(self, X_valid, y_valid, valid_precision):
        train_loss = self.calculate_loss(self.kernel_params)
        return train_loss, np.zeros((1, 1))

    def log_marg_likelihood_old(self, y, precision, cov, nu=None, tau=None):

        if nu is None:
            nu = tf.identity(self.nu)
        if tau is None:
            tau = tf.identity(self.tau)

        S_root = tf.linalg.diag(tf.squeeze(tau)) ** (1 / 2)
        B0 = tf.linalg.matmul(S_root, cov)
        B = tf.eye(y.shape[0], dtype=tf.float64) + tf.linalg.matmul(B0, S_root)
        L = tf.linalg.cholesky(B)

        V = tf.linalg.solve(tf.transpose(L), B0)
        post_cov = cov - tf.linalg.matmul(tf.transpose(V), V)
        mu = tf.linalg.matmul(post_cov, nu)

        tau_cavity = 1 / tf.expand_dims(tf.linalg.diag_part(post_cov), axis=-1) - tau
        nu_cavity = (1 / tf.expand_dims(tf.linalg.diag_part(post_cov), axis=-1)) * mu - nu

        T1 = 0.5 * tf.reduce_sum(tf.math.log(1 + (tau / tau_cavity))) \
             - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        st_inv = tf.linalg.diag(tf.squeeze(1 / (tau_cavity + tau)))
        T2 = 0.5 * tf.linalg.matmul(tf.transpose(nu), tf.linalg.matmul(post_cov - st_inv, nu))

        mu_cavity = nu_cavity / tau_cavity
        T3 = 0.5 * tf.linalg.matmul(tf.transpose(mu_cavity),
                                    tf.linalg.matmul(tf.linalg.diag(tf.squeeze(tau_cavity)) * st_inv,
                                                     tau * mu_cavity - 2 * nu))

        z = y * mu_cavity / tf.math.sqrt(1 + (1 / tau_cavity))
        z_hat = (2 * precision - 1) * scipy_norm.cdf(z) + 1 - precision
        T4 = tf.reduce_sum(tf.math.log(z_hat))# + 0.5 * tf.reduce_sum(tf.math.log((1 / tau_cavity) + (1 / tau))) \
             #+ 0.5 * tf.reduce_sum((mu_cavity - (nu / tau)) / ((1 / tau_cavity) + (1 / tau)))

        return T1 + T2 + T3 + T4

    def log_marg_likelihood(self, X, y, precision, cov, nu=None, tau=None, scale_precision=False):

        if scale_precision:
            precision = self.scale_precision(X, precision)

        if nu is None:
            nu = tf.identity(self.nu)
        if tau is None:
            tau = tf.identity(self.tau)

        S_root = tf.linalg.diag(tf.squeeze(tau)) ** (1 / 2)
        B0 = tf.linalg.matmul(S_root, cov)
        B = tf.eye(y.shape[0], dtype=tf.float64) + tf.linalg.matmul(B0, S_root)
        L = tf.linalg.cholesky(B)

        V = tf.linalg.solve(tf.transpose(L), B0)
        post_cov = cov - tf.linalg.matmul(tf.transpose(V), V)
        mu = tf.linalg.matmul(post_cov, nu)

        tau_cavity = 1 / tf.expand_dims(tf.linalg.diag_part(post_cov), axis=-1) - tau
        nu_cavity = (1 / tf.expand_dims(tf.linalg.diag_part(post_cov), axis=-1)) * mu - nu

        T1 = 0.5 * tf.reduce_sum(tf.math.log(1 + (tau / tau_cavity))) \
             + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))  # CHANGE OF SIGN FROM RASMUSSEN & WILLIAMS

        st_inv = tf.linalg.diag(tf.squeeze(1 / (tau_cavity + tau)))

        T2 = 0.5 * tf.linalg.matmul(tf.transpose(nu), tf.linalg.matmul(2 * tf.linalg.diag(tf.squeeze(1 / tau))
                                                                       - st_inv - post_cov, nu))  # CHANGE OF SIGN FROM RASMUSSEN & WILLIAMS

        mu_cavity = nu_cavity / tau_cavity
        T3 = 0.5 * tf.linalg.matmul(tf.transpose(mu_cavity),
                                    tf.linalg.matmul(tf.linalg.diag(tf.squeeze(tau_cavity)) * st_inv,
                                                     tau * mu_cavity - 2 * nu))

        z = y * mu_cavity / tf.math.sqrt(1 + (1 / tau_cavity))
        z_hat = (2 * precision - 1) * scipy_norm.cdf(z) + 1 - precision
        T4 = tf.reduce_sum(tf.math.log(z_hat))# + 0.5 * tf.reduce_sum(tf.math.log((1 / tau_cavity) + (1 / tau))) \
             #+ 0.5 * tf.reduce_sum((mu_cavity - (nu / tau)) / ((1 / tau_cavity) + (1 / tau)))

        return T1 + T2 + T3 + T4

    def calculate_loss(self, kp):
        cov = self._calculate_covariance_spec(kp)
        return - self.log_marg_likelihood(self.X_train, self.y_train, self.scaled_precision, cov)

    def calculate_loss_new_data(self, X_new, y_new, precision_new, kp):
        scaled_precision = self.scale_precision(X_new, precision_new)
        return None

    def single_posterior_predict(self, X_new):
        """Get predictive mean and variance (Alg. 3.6 in Rasmussen and Williams (2006))"""
        X_new = tf.convert_to_tensor(X_new)
        k = tf.expand_dims(self.kernel(X_new, self.X_train), axis=-1)
        c = self.kernel(X_new, X_new)

        m = tf.linalg.matmul(tf.transpose(k), (self.nu - self.z))

        S_root = tf.linalg.diag(tf.squeeze(self.tau))**(1/2)
        v = tf.linalg.solve(self.L, tf.linalg.matmul(S_root, k))
        s = c - tf.linalg.matmul(tf.transpose(v), v)

        return m.numpy(), s.numpy()

    def single_predict(self, X_new):
        m, s = self.single_posterior_predict(X_new)
        return scipy_norm.cdf(m / np.sqrt(1 + s))

    def predict(self, X_new):
        m, s = self.posterior_predict(X_new)

        return scipy_norm.cdf(m / np.sqrt(1 + s))

    def posterior_predict(self, X_new, return_std=True):
        """Get predictive mean and variance (Alg. 3.6 in Rasmussen and Williams (2006))"""
        X_new = tf.convert_to_tensor(X_new)
        k = self.kernel(tf.reshape(X_new, [1, X_new.shape[0], X_new.shape[-1]]),
                        tf.reshape(self.X_train, [self.X_train.shape[0], 1, self.X_train.shape[-1]]))
        c = self.kernel(X_new, X_new)

        m = tf.linalg.matmul(tf.transpose(k), (self.nu - self.z))

        if return_std:
            S_root = tf.linalg.diag(tf.squeeze(self.tau)) ** (1 / 2)
            v = tf.linalg.solve(self.L, tf.linalg.matmul(S_root, k))

            # TODO: could calculate only diagonal
            s = c - tf.linalg.diag_part(tf.linalg.matmul(tf.transpose(v), v))

            return m.numpy(), s.numpy()[:, np.newaxis]
        else:
            return m.numpy()

    def noisy_predict(self, X_new, precision):
        prob_y = self.predict(X_new)

        return self.prob_noisy_y_given_prob_y(X_new, prob_y, precision)

    def predict_model_var(self, X_new):
        X_new = tf.convert_to_tensor(X_new)
        k = self.kernel(tf.reshape(X_new, [1, X_new.shape[0], X_new.shape[-1]]),
                         tf.reshape(self.X_train, [self.X_train.shape[0], 1, self.X_train.shape[-1]]))
        c = self.kernel(X_new, X_new)

        S_root = tf.linalg.diag(tf.squeeze(self.tau)) ** (1 / 2)
        v = tf.linalg.solve(self.L, tf.linalg.matmul(S_root, k))
        s = c - tf.linalg.diag_part(tf.linalg.matmul(tf.transpose(v), v))

        return s.numpy()[:, np.newaxis]

    def kernel(self, x_m, x_n):
        return self._kernel(x_m, x_n, self.kernel_params)

    def prob_noisy_y_given_prob_y(self, X, prob_y, precision=1.0):
        scaled_precision = self.scale_precision(X, precision)

        return ((2 * scaled_precision - 1) * prob_y + (1 - scaled_precision)).numpy()

