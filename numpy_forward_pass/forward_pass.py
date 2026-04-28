import numpy as np
from activation_fn import relu, sigmoid

def mlp_forward(input, params):
    w1, b1 = params['w1'], params['b1']
    w2, b2 = params['w2'], params['b2']

    z1 = input @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)

    return a2




batch_size = 32
input_dim = 3
hidden_dim = 5
output_dim = 1
input = np.random.randn(batch_size, input_dim)

params = {
    'w1': np.random.randn(input_dim, hidden_dim),
    'b1': np.zeros(hidden_dim),
    'w2': np.random.randn(hidden_dim, output_dim),
    'b2': np.zeros(output_dim)
}

output = mlp_forward(input, params)
print(output)
