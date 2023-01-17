import random

import matplotlib.pyplot as plt
import numpy as np

# Constants
N_ITERS = 1500
LR = 0.01
N_POINTS = 100
SLOPE = 0.25  # True slope of line of best fit to noisy data
INTERCEPT = 1  # True intercept of line of best fit to noisy data
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class LinearRegression:
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

    def error(self, y, y_pred):
        dy = y - y_pred
        error = sum([_dy**2 for _dy in dy])
        return error

    def compute_grad_w(self, y, y_pred, x):
        return sum((2 * (y_pred - y) * x)[0]) / len(x)

    def compute_grad_b(self, y, y_pred):
        return sum((2 * (y_pred - y) * 1)) / len(x)

    def fit(self, x, y, n_iters=100, learning_rate=0.01):
        self.x = np.atleast_2d(x)
        self.y = np.array(y)
        self.init_weights()

        for _ in range(n_iters):
            self.pred = self.predict(x)
            error = self.error(self.y, self.pred)

            print(error)

            grad_w = self.compute_grad_w(self.y, self.pred, self.x)
            grad_b = self.compute_grad_b(self.y, self.pred)
            self.W = self.W - learning_rate * grad_w
            self.b = self.b - learning_rate * grad_b

    def predict(self, x):
        return self.W * x + self.b


def noise(rg, scale=0.15):
    return scale * (2 * rg.random() - 1)


def get_noisy_data(n_points, rg, actual_slope, actual_intercept):
    x = [n / n_points for n in range(n_points)]
    y = [actual_intercept + actual_slope * (_x + noise(rg)) for _x in x]
    return np.array(x), np.array(y)


if __name__ == "__main__":
    model = LinearRegression()
    x, y = get_noisy_data(N_POINTS, random.Random(), SLOPE, INTERCEPT)
    plt.scatter(x, y)
    model.fit(x, y, n_iters=N_ITERS, learning_rate=LR)
    y_pred = model.predict(x)

    plt.scatter(x, y_pred)
    plt.show()
