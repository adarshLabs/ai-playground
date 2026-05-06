# AI Playground

A small NumPy-first playground for learning neural-network building blocks.
The repository contains compact implementations of MLP forward/backward passes,
normalization layers, and regularization helpers, with unit tests for the core
math.

## Project Structure

```text
.
├── mlp/
│   ├── activation_functions.py
│   ├── backpropagation.py
│   ├── forward_pass.py
│   └── xor_problem.py
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

`torch` is only needed for the comparison demo in `mlp/forward_pass.py`; the
tested layer and regularization helpers use NumPy.

## Run Tests

```bash
python -m unittest discover -s tests
```

## Run Examples

```bash
python -m mlp.forward_pass
python -m mlp.backpropagation
python -m mlp.xor_problem
```

## Modules

- `mlp`: simple MLP examples, activation functions, forward pass comparison,
  backpropagation, and XOR training.
- `normalization`: batch normalization and layer normalization implementations.
- `regularization`: dropout plus L1/L2 loss and gradient helpers.
