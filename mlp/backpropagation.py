import numpy as np


from .forward_pass import np_forward



def backpropagation(Y, cache, params):
    X, z1, a1, z2, a2 = cache
    N = X.shape[0]
    w2 = params['w2']

    dz2 = (a2 - Y)/N
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2)

    da1 = dz2 @ w2.T
    dz1 = da1 * (z1> 0)
    dw1 = X.T @ dz1
    db1 = np.sum(dz1)

    grads = {
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2
    }
    return grads


def steps(X, Y, params, lr=0.01):
    y_pred, cache = np_forward(X, params)
    grads = backpropagation(Y, cache, params)

    params['w1'] -= lr* grads['dw1']
    params['b1'] -= lr* grads['db1']
    params['w2'] -= lr* grads['dw2']
    params['b2'] -= lr* grads['db2']

    return params


def main():
    batch_size = 32
    input_dim = 3
    hidden_dim = 5
    output_dim = 1

    np.random.seed(42)
    inputs = np.random.randn(batch_size, input_dim)
    targets = np.random.randn(batch_size, output_dim)

    params = {
        'w1': np.random.randn(input_dim, hidden_dim),
        'b1': np.random.randn(hidden_dim),
        'w2': np.random.randn(hidden_dim, output_dim),
        'b2': np.random.randn(output_dim)
    }

    y1, _ = np_forward(inputs, params)

    print(sum(y1 - targets))
    for i in range(3000):
        params = steps(inputs, targets, params)
        if i % 100 == 0:
            y2, _ = np_forward(inputs, params)
            print(i, sum(y2 - targets))


if __name__ == "__main__":
    main()

