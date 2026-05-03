import numpy as np
class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=.9):
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.random.randn((1, dim))
        self.beta = np.random.randn((1, dim))

        self.running_mean = np.random.randn((1, dim))
        self.running_var = np.random.randn((1, dim))

    
    def forward(self, X, training=True):

        if training:
            mu =  np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keep_dims=True)

            x_centered = (X-mu)
            std_inv = 1.0 / np.sqrt(var + self.eps)

            x_hat = x_centered * std_inv


            self.running_mean = self.running_mean * self.momentum + (1-self.momentum) * mu
            self.running_var = self.running_var * self.momentum + (1-self.momentum)* var

            self.cache = (x_hat, std_inv, x_centered, var)

        else:
            x_hat = (X - self.running_mean) / np.sqrt(self.running_mean + self.eps)

        out = x_hat * self.gamma + self.beta
        return out
    






