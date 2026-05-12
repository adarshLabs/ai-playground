import numpy as np

import torch


from mlp.numpy.activation_functions import relu, sigmoid


def np_forward(inputs, params):
    w1, b1 = params['w1'], params['b1']
    w2, b2 = params['w2'], params['b2']

    z1 = inputs @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    cache = (inputs, z1, a1, z2, a2)
    return a2, cache


def torch_forward(inputs, params):
    if torch is None:
        raise ImportError("torch is required to run torch_forward")

    w1, b1 = params['w1'], params['b1']
    w2, b2 = params['w2'], params['b2']

    z1 = inputs @ w1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ w2 + b2
    a2 = torch.sigmoid(z2)

    return a2


def main():
    np.random.seed(42)

    batch_size = 32
    input_dim = 3
    hidden_dim = 5
    output_dim = 1
    input_np = np.random.randn(batch_size, input_dim)
    input_t = torch.tensor(input_np, dtype=torch.float32)

    params_np = {
        'w1': np.random.randn(input_dim, hidden_dim),
        'b1': np.random.randn(hidden_dim),
        'w2': np.random.randn(hidden_dim, output_dim),
        'b2': np.random.randn(output_dim)
    }
    params_t =  {
        'w1': torch.tensor(params_np['w1'], dtype=torch.float32), 
        'b1': torch.tensor(params_np['b1'], dtype=torch.float32),
        'w2': torch.tensor(params_np['w2'], dtype=torch.float32), 
        'b2': torch.tensor(params_np['b2'], dtype=torch.float32)
    }

    output_np, _ = np_forward(input_np, params_np)
    output_t = torch_forward(input_t, params_t).detach().numpy()

    result = np.allclose(output_np, output_t, atol=1e-6)
    print(result, output_np, output_t)


if __name__ == "__main__":
    main()
