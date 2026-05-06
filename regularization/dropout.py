import numpy as np

class Dropout:
    def __init__(self, drop_prob= 0.5, seed=None):

        assert 0<=drop_prob<1
        self.keep_prob= 1-drop_prob
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, X, training=True):

        if not training or self.keep_prob ==0:
            return X
        
        self.mask = self.rng.random(X.shape) < self.keep_prob
        X = (X * self.mask)/self.keep_prob

        return X
    

    def backward(self, dout):
        if self.mask is None or self.keep_prob==0 :
            return dout
        
        dx = (dout * self.mask) / self.keep_prob

        return dx
