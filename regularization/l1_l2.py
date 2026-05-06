import numpy as np


def l1_loss(W, lambda_):
    return lambda_ * np.sum(abs(W))

def l1_grad(W, lambda_):
    return lambda_ * np.sign(W)


def l2_loss(W, lambda_):
    return 0.5 * lambda_ * np.sum(W**2)

def l2_grad(W, lambda_):
    return lambda_ * W

