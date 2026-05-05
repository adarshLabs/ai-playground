import unittest

import numpy as np

from normalisation.layer_normalisation import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def test_forward_normalises_each_sample(self):
        x = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
            ]
        )
        layer_norm = LayerNorm(dim=4, eps=1e-5)

        out = layer_norm.forward(x)

        np.testing.assert_allclose(np.mean(out, axis=1), np.zeros(2), atol=1e-7)
        np.testing.assert_allclose(np.var(out, axis=1), np.ones(2), rtol=1e-5)

    def test_forward_applies_learnable_scale_and_shift(self):
        x = np.array([[1.0, 2.0, 3.0]])
        layer_norm = LayerNorm(dim=3, eps=1e-5)
        layer_norm.gamma = np.array([[1.0, 2.0, 3.0]])
        layer_norm.beta = np.array([[0.5, -1.0, 2.0]])

        out = layer_norm.forward(x)

        mu = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        expected = ((x - mu) / np.sqrt(var + layer_norm.eps)) * layer_norm.gamma + layer_norm.beta
        np.testing.assert_allclose(out, expected)

    def test_backward_matches_numerical_gradient(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(3, 4))
        dout = rng.normal(size=(3, 4))
        layer_norm = LayerNorm(dim=4, eps=1e-5)
        layer_norm.gamma = rng.normal(size=(1, 4))
        layer_norm.beta = rng.normal(size=(1, 4))

        layer_norm.forward(x)
        dx, dgamma, dbeta = layer_norm.backward(dout)
        expected_dgamma = np.sum(dout * layer_norm.x_hat, axis=0, keepdims=True)
        expected_dbeta = np.sum(dout, axis=0, keepdims=True)

        np.testing.assert_allclose(dx, self._numerical_dx(layer_norm, x, dout), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dgamma, expected_dgamma)
        np.testing.assert_allclose(dbeta, expected_dbeta)

    def _numerical_dx(self, layer_norm, x, dout, h=1e-5):
        dx = np.zeros_like(x)

        for index in np.ndindex(x.shape):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[index] += h
            x_minus[index] -= h

            out_plus = layer_norm.forward(x_plus)
            out_minus = layer_norm.forward(x_minus)
            loss_plus = np.sum(out_plus * dout)
            loss_minus = np.sum(out_minus * dout)
            dx[index] = (loss_plus - loss_minus) / (2 * h)

        return dx


if __name__ == "__main__":
    unittest.main()
