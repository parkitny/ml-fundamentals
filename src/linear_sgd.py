import random

import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    def __init__(self, rand_generator):
        self.x = None
        self.y = None
        self.preds = None
        self.rg = rand_generator

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


if __name__ == "__main__":
    rand_generator = random.Random()
    model = LinearRegression(rand_generator)
    n_points = 100
    actual_slope = 0.25
    actual_intercept = 1
    x = [n / n_points for n in range(n_points)]
    scale = 0.15
    y = [
        actual_intercept
        + actual_slope * (_x + scale * (2 * rand_generator.random() - 1))
        for _x in x
    ]
    plt.scatter(x, y)
    model.fit(np.array(x), np.array(y), n_iters=1500)
    y_pred = model.predict(x)

    plt.scatter(list(x), list(y_pred))
    plt.show()
