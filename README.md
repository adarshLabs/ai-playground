# AI Playground

A small NumPy-first playground for learning neural-network building blocks.
The repository contains compact implementations of MLP forward/backward passes,
normalization layers, and regularization helpers, with unit tests for the core
math. It also includes a PyTorch XOR example for comparing a hand-written
training loop with a framework-based version.

## Project Structure

```text
.
├── mlp/
│   ├── numpy/
│   │   ├── activation_functions.py
│   │   ├── backpropagation.py
│   │   ├── forward_pass.py
│   │   └── xor_problem.py
│   └── pytorch/
│       ├── custom_autograd.py
│       └── xor.py
├── normalization/
│   ├── batch_normalization.py
│   └── layer_normalization.py
├── regularization/
│   ├── dropout.py
│   └── l1_l2.py
└── tests/
    ├── test_batch_normalization.py
    ├── test_dropout.py
    ├── test_l1_l2.py
    └── test_layer_normalization.py
```

## Setup

Create and activate a virtual environment, then install the runtime
dependencies used by the examples:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch
```

`torch` is only needed for the PyTorch demos in `mlp/numpy/forward_pass.py`
and `mlp/pytorch/`; the tested layer and regularization helpers use NumPy.

## Run Tests

```bash
python -m unittest discover -s tests
```

## Run Examples

```bash
python -m mlp.numpy.forward_pass
python -m mlp.numpy.backpropagation
python -m mlp.numpy.xor_problem
python -m mlp.pytorch.xor
python -m mlp.pytorch.custom_autograd
```

## Modules

- `mlp.numpy`: simple NumPy MLP examples, activation functions, forward pass
  comparison, backpropagation, and NumPy XOR training.
- `mlp.pytorch`: PyTorch XOR training and custom autograd examples.
- `normalization`: batch normalization and layer normalization implementations.
- `regularization`: dropout plus L1/L2 loss and gradient helpers.

## Learning Path

1. Start with `mlp/numpy/activation_functions.py` for the basic nonlinearities.
2. Run `mlp/numpy/forward_pass.py` to compare a manual NumPy forward pass with
   PyTorch.
3. Read `mlp/numpy/backpropagation.py` and `mlp/numpy/xor_problem.py` to follow
   gradient updates by hand.
4. Run `mlp/pytorch/xor.py` to see the same XOR task expressed with
   `torch.nn` modules, `BCELoss`, and `optim.SGD`.
