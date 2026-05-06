import numpy as np


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps

        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))

    def forward(self, x, training=True):
        self.x = x
        self.mu = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)

        x_centered = x - self.mu
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.x_hat = x_centered * std_inv

        out = self.x_hat * self.gamma + self.beta

        return out

    def backward(self, dout):

        _, D = dout.shape
        dgamma = np.sum(dout * self.x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dx_hat = dout * self.gamma
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        x_centered = self.x - self.mu

        dx = (1.0 / D) * std_inv * (
            D * dx_hat
            - np.sum(dx_hat, axis=1, keepdims=True)
            - x_centered
            * std_inv**2
            * np.sum(dx_hat * x_centered, axis=1, keepdims=True)
        )

        return dx, dgamma, dbeta
