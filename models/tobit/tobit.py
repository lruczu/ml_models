import numpy as np
import tensorflow as tf


class Tobit:
    """
    This class represents popular Tobit model as described,
    for example: https://en.wikipedia.org/wiki/Tobit_model.

    Optimization is done through maximizing likelihood function (mle)
    """
    def __init__(
        self,
        cut_off: float=0,
        learning_rate: float=1e-03,
        max_iter=1000,
        tol: float=1e-09
    ):
        """
        parameters
        ----------
        cut_off: the smallest observable value of latent variable
        learning_rate: learning rate for Adam optimizer
        max_iter: maximum number of epochs
        tol: absolute change between adjacent losses that stops fitting
        """
        self.cut_off = cut_off
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        self.sess = None

    def fit(self, X, y):
        self._check_args(X, y)
        X, y = self._reshape_args(X, y)

        self.sess = tf.Session(graph=self._build_graph(X.shape[1]))
        self.sess.run(self.init)

        history = []
        with self.sess.as_default() as sess:
            prev_loss = -np.inf
            for i in range(self.max_iter):
                _, loss = sess.run(self.train_step, feed_dict={
                    self.x: X,
                    self.y: y
                })
                history.append(loss)

                if np.isinf(prev_loss):
                    prev_loss = loss
                    continue

                change = np.abs(loss - prev_loss)
                if change < self.tol:
                    print('Convergence attained after {} epochs'.format(i+1))
                    return history
                prev_loss = loss
            result = 1 * (change > self.tol)
            print('After {} epochs, the optimization did {} converge, absolute change: {}'.
                  format(self.max_iter, ['not', ''][result], change))
            return history

    def predict(self, X):
        X = self._reshape_args(X)
        if self.sess is None:
            raise ValueError('Model must be fitted first!')

        with self.sess.as_default() as sess:
            return sess.run(self.output, feed_dict={
                self.x: X
            })

    def _build_graph(self, n_features: int):
        with tf.Graph().as_default() as graph:
            with tf.name_scope('input'):
                self.x = tf.placeholder(shape=[None, n_features], dtype=tf.float32)
                self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            with tf.name_scope('parameters'):
                self.w = tf.Variable(tf.random_normal((n_features, 1)), dtype=tf.float32)
                self.b = tf.Variable(0, dtype=tf.float32)
                self.log_var = tf.Variable(10, dtype=tf.float32)
                self.sigma = tf.sqrt(tf.exp(self.log_var)) + 1e-03

            with tf.name_scope('output'):
                self.y_hat = tf.matmul(self.x, self.w) + self.b
                self.output = tf.maximum(self.y_hat, self.cut_off)

            gaussian = tf.distributions.Normal(loc=0.0, scale=1.0)

            # 1 term, y* > self.cut_off
            term1 = tf.log(1 / self.sigma * gaussian.prob((self.y - self.y_hat) / self.sigma) + 1e-03)
            term1_mask = 1 - tf.cast(tf.equal(self.y, self.cut_off), dtype=tf.float32)

            # 2 term, y * == self.cut_off
            term2 = tf.log(1 - gaussian.cdf((self.y_hat - self.cut_off) / self.sigma) + 1e-03)
            term2_mask = 1 - term1_mask

            mle = term1 * term1_mask + term2 * term2_mask

            self.loss = -tf.reduce_mean(mle)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            grads = opt.compute_gradients(self.loss)
            for g, v in grads:
                tf.check_numerics(g, message=v.name)

            self.train_step = (opt.apply_gradients(grads), self.loss)

            self.init = tf.global_variables_initializer()

            return graph

    def sigma_(self):
        if self.sess:
            with self.sess.as_default() as sess:
                return sess.run(self.sigma) - 1e-03

    def bias_(self):
        if self.sess:
            with self.sess.as_default() as sess:
                return sess.run(self.b)

    def weights_(self):
        if self.sess:
            with self.sess.as_default() as sess:
                return sess.run(self.w)

    def _check_args(self, X, y):
        if np.any(np.isnan(y)):
            raise ValueError('vector y has missing values!')
        if np.any(np.isnan(X)):
            raise ValueError('matrix X has missing values!')

        if np.min(y) < self.cut_off:
            raise ValueError('vector y has smaller value than {}!'.format(self.cut_off))

    def _reshape_args(self, X, y=None):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        if y is not None:
            if len(y.shape) == 1:
                y = y[:, np.newaxis]
            return X, y

        return X
