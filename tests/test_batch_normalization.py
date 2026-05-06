import unittest

import numpy as np



from normalization.batch_normalization import BatchNorm


class TestBatchNorm(unittest.TestCase):
    def test_forward_normalises_each_feature_during_training(self):
        x = np.array(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
            ]
        )
        batch_norm = BatchNorm(dim=3, eps=1e-5)

        out = batch_norm.forward(x, training=True)

        np.testing.assert_allclose(np.mean(out, axis=0), np.zeros(3), atol=1e-7)
        np.testing.assert_allclose(np.var(out, axis=0), np.ones(3), rtol=1e-5)

    def test_forward_updates_running_statistics(self):
        x = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        batch_norm = BatchNorm(dim=2, momentum=0.8)

        batch_norm.forward(x, training=True)

        expected_mean = 0.8 * np.zeros((1, 2)) + 0.2 * np.mean(x, axis=0, keepdims=True)
        expected_var = 0.8 * np.ones((1, 2)) + 0.2 * np.var(x, axis=0, keepdims=True)
        np.testing.assert_allclose(batch_norm.running_mean, expected_mean)
        np.testing.assert_allclose(batch_norm.running_var, expected_var)

    def test_forward_uses_running_statistics_during_inference(self):
        x = np.array([[10.0, 20.0], [30.0, 40.0]])
        batch_norm = BatchNorm(dim=2, eps=1e-5)
        batch_norm.running_mean = np.array([[5.0, 10.0]])
        batch_norm.running_var = np.array([[25.0, 100.0]])
        batch_norm.gamma = np.array([[2.0, 3.0]])
        batch_norm.beta = np.array([[1.0, -1.0]])

        out = batch_norm.forward(x, training=False)

        expected = (x - batch_norm.running_mean) / np.sqrt(batch_norm.running_var + batch_norm.eps)
        expected = expected * batch_norm.gamma + batch_norm.beta
        np.testing.assert_allclose(out, expected)

    def test_backward_matches_numerical_gradient(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(5, 4))
        dout = rng.normal(size=(5, 4))
        batch_norm = BatchNorm(dim=4, eps=1e-5)
        batch_norm.gamma = rng.normal(size=(1, 4))
        batch_norm.beta = rng.normal(size=(1, 4))

        batch_norm.forward(x, training=True)
        dx, dgamma, dbeta = batch_norm.backward(dout)
        x_hat = batch_norm.cache[0]
        expected_dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        expected_dbeta = np.sum(dout, axis=0, keepdims=True)

        np.testing.assert_allclose(dx, self._numerical_dx(batch_norm, x, dout), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dgamma, expected_dgamma)
        np.testing.assert_allclose(dbeta, expected_dbeta)

    def _numerical_dx(self, batch_norm, x, dout, h=1e-5):
        dx = np.zeros_like(x)

        for index in np.ndindex(x.shape):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[index] += h
            x_minus[index] -= h

            out_plus = batch_norm.forward(x_plus, training=True)
            out_minus = batch_norm.forward(x_minus, training=True)
            loss_plus = np.sum(out_plus * dout)
            loss_minus = np.sum(out_minus * dout)
            dx[index] = (loss_plus - loss_minus) / (2 * h)

        return dx


if __name__ == "__main__":
    unittest.main()
