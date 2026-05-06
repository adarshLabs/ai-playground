import numpy as np


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu_grad(x):
    return (x > 0).astype(float)

def binary_cross_entropy_loss(Y, y_pred):
    eps = 1e-8
    y_pred_clipped = np.clip(y_pred, eps, 1-eps)
    return -np.mean(Y * np.log(y_pred_clipped) + (1-Y) * np.log(1 - y_pred_clipped))

def forward(X, params):
    w1, b1, w2, b2 = params['w1'], params['b1'], params['w2'], params['b2']

    z1 = X @ w1 + b1
    a1 = relu(z1)

    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)

    cache = (X, z1, a1, z2, a2)
    return a2, cache

def backpropagation(Y, cache, params):
    X, z1, a1, z2, a2 = cache
    w2 = params['w2']
    N = X.shape[0]

    dz2 = (a2 - Y)/N
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)/N

    da1 = dz2 @ w2.T
    dz1 = da1 * (z1>0)
    dw1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)/N

    grads = {
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2, 
        'db2': db2
    }
    
    return grads



def main():
    input_dim= 2
    hidden_dim =4
    output_dim = 1
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    params = {
        'w1': np.random.randn(input_dim, hidden_dim),
        'b1': np.zeros(hidden_dim),
        'w2': np.random.randn(hidden_dim, output_dim),
        'b2': np.zeros(output_dim)
    }

    epochs = 10000
    lr = 0.1
    for i in range(epochs+1):
        y_pred, cache = forward(X, params)
        
        loss = binary_cross_entropy_loss(Y, y_pred)
        
        grads = backpropagation(Y, cache, params)

        params['w1'] -= lr * grads['dw1']
        params['b1'] -= lr * grads['db1']
        params['w2'] -= lr * grads['dw2']
        params['b2'] -= lr * grads['db2']

        if i%1000==0:
            print(f"EPOCH {i}: {loss}")
    
    y_new, _ = forward(X, params)
    pred = (y_new>0.5).astype(int)
    for r, s in zip (X, pred):
        print(r, s)


if __name__=="__main__":
    main()
