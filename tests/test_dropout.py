import unittest

import numpy as np

from Regularisation.dropout import Dropout


class TestDropout(unittest.TestCase):
    def test_forward_drops_and_scales_values_during_training(self):
        x = np.ones((3, 4))
        dropout = Dropout(drop_prob=0.25, seed=123)
        expected_rng = np.random.default_rng(123)
        expected_mask = expected_rng.random(x.shape) < 0.75

        out = dropout.forward(x, training=True)

        np.testing.assert_array_equal(dropout.mask, expected_mask)
        np.testing.assert_allclose(out, (x * expected_mask) / 0.75)

    def test_forward_returns_input_during_inference(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        dropout = Dropout(drop_prob=0.5, seed=7)

        out = dropout.forward(x, training=False)

        self.assertIs(out, x)
        self.assertIsNone(dropout.mask)

    def test_backward_applies_forward_mask_and_scaling(self):
        x = np.ones((2, 3))
        dout = np.arange(6.0).reshape(2, 3)
        dropout = Dropout(drop_prob=0.5, seed=42)

        dropout.forward(x, training=True)
        dx = dropout.backward(dout)

        expected = (dout * dropout.mask) / 0.5
        np.testing.assert_allclose(dx, expected)

    def test_backward_without_forward_returns_upstream_gradient(self):
        dout = np.array([[1.0, -2.0], [3.0, -4.0]])
        dropout = Dropout(drop_prob=0.5)

        dx = dropout.backward(dout)

        self.assertIs(dx, dout)

    def test_drop_probability_must_be_between_zero_and_one(self):
        with self.assertRaises(AssertionError):
            Dropout(drop_prob=-0.1)

        with self.assertRaises(AssertionError):
            Dropout(drop_prob=1.0)


if __name__ == "__main__":
    unittest.main()
