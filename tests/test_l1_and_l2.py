import unittest

import numpy as np

from Regularisation.l1_and_l2 import l1_grad, l1_loss, l2_grad, l2_loss


class TestL1AndL2Regularisation(unittest.TestCase):
    def test_l1_loss_sums_absolute_weights(self):
        weights = np.array([[1.0, -2.0, 0.0], [3.5, -4.5, 2.0]])

        loss = l1_loss(weights, lambda_=0.2)

        self.assertEqual(loss, 0.2 * 13.0)

    def test_l1_grad_returns_scaled_signs(self):
        weights = np.array([[1.0, -2.0, 0.0], [3.5, -4.5, 2.0]])

        grad = l1_grad(weights, lambda_=0.2)

        expected = np.array([[0.2, -0.2, 0.0], [0.2, -0.2, 0.2]])
        np.testing.assert_allclose(grad, expected)

    def test_l2_loss_sums_half_squared_weights(self):
        weights = np.array([[1.0, -2.0, 0.0], [3.0, -4.0, 2.0]])

        loss = l2_loss(weights, lambda_=0.5)

        self.assertEqual(loss, 0.5 * 0.5 * 34.0)

    def test_l2_grad_returns_scaled_weights(self):
        weights = np.array([[1.0, -2.0, 0.0], [3.0, -4.0, 2.0]])

        grad = l2_grad(weights, lambda_=0.5)

        np.testing.assert_allclose(grad, 0.5 * weights)

    def test_zero_lambda_removes_regularisation_penalty(self):
        weights = np.array([[1.0, -2.0], [3.0, -4.0]])

        self.assertEqual(l1_loss(weights, lambda_=0.0), 0.0)
        self.assertEqual(l2_loss(weights, lambda_=0.0), 0.0)
        np.testing.assert_allclose(l1_grad(weights, lambda_=0.0), np.zeros_like(weights))
        np.testing.assert_allclose(l2_grad(weights, lambda_=0.0), np.zeros_like(weights))


if __name__ == "__main__":
    unittest.main()
