import numpy as np
class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=.9):
        self.eps = eps
        self.momentum = momentum

        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))

        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))

    
    def forward(self, X, training=True):

        if training:
            mu =  np.mean(X, axis=0, keepdims=True)
            var = np.var(X, axis=0, keepdims=True)

            x_centered = (X-mu)
            std_inv = 1.0 / np.sqrt(var + self.eps)

            x_hat = x_centered * std_inv


            self.running_mean = self.running_mean * self.momentum + (1-self.momentum) * mu
            self.running_var = self.running_var * self.momentum + (1-self.momentum)* var

            self.cache = (x_hat, std_inv, x_centered, var)

        else:
            x_hat = (X - self.running_mean) / np.sqrt(self.running_var+ self.eps)

        out = x_hat * self.gamma + self.beta
        return out
    
    def backward(self, dout):
        x_hat, std_inv, x_centered, var = self.cache

        N, D = dout.shape

        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)

        dx_hat = dout * self.gamma
        dstd_inverse = x_centered * dx_hat
        dvar = np.sum(dstd_inverse * -0.5 * std_inv**3, axis = 0, keepdims=True)
        dmu = np.sum(-dx_hat * std_inv, axis=0, keepdims=True) + dvar * np.mean(-2* x_centered, axis=0, keepdims=True) 
        dx = dx_hat * std_inv + 2* dvar * x_centered/N + dmu/N
        
        return dx, dgamma, dbeta






