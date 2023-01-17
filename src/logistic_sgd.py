import random

import matplotlib.pyplot as plt
import numpy as np

# Constants
N_ITERS = 10000
LR = 0.1
N_POINTS = 100
SPLIT = 0.65  # 1 / 0 label split point
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class LogisticRegression:
    def __init__(self):
        self.x = None
        self.y = None
        self.preds = None
        self.rg = random.Random()

    def _rand(self, i):
        return self.rg.random()

    def init_weights(self):
        self.W = np.zeros(self.x.shape[0])
        self.W = np.vectorize(self._rand)(self.W)
        self.b = np.zeros(self.x.shape[0])
        self.b = np.vectorize(self._rand)(self.b)

    def loss(self, y, y_pred):  # Negative log likelihood.
        log_loss = -1 * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return sum(log_loss)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_grad_w(self, y, y_pred, x):
        difference = y_pred - y
        return np.dot(x, difference) / len(y)

    def compute_grad_b(self, y, y_pred):
        difference = y_pred - y
        return sum(difference) / len(y)

    def fit(self, x, y, n_iters=100, learning_rate=0.01):
        self.x = np.atleast_2d(x)
        self.y = np.array(y)
        self.init_weights()

        for _ in range(n_iters):
            self.pred = self.predict(x)
            error = self.loss(self.y, self.pred)

            print(error)

            grad_w = self.compute_grad_w(self.y, self.pred, self.x)
            grad_b = self.compute_grad_b(self.y, self.pred)
            self.W = self.W - learning_rate * grad_w
            self.b = self.b - learning_rate * grad_b

    def predict(self, x):
        z = self.W * x + self.b
        return self.sigmoid(z)


def get_data(n_points, split):
    x = np.array([n / n_points for n in range(n_points)])
    y = np.zeros_like(x)
    y[: int(len(y) * split)] = 1
    return x, y


if __name__ == "__main__":
    model = LogisticRegression()
    x, y = get_data(N_POINTS, SPLIT)
    plt.scatter(x, y)
    model.fit(x, y, n_iters=N_ITERS, learning_rate=LR)
    y_pred = model.predict(x)

    plt.scatter(x, y_pred)
    plt.show()
