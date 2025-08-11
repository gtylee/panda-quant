import unittest
import numpy as np

from rbfi_approximator import RadialBasisFunctionInterpolator


class TestRBFIRoundtrip(unittest.TestCase):
    def test_to_from_dict(self):
        centers = np.array([[0.0, 0.0], [1.0, 1.0]])
        weights = np.array([0.5, 1.5])
        length_scales = np.array([0.3, 0.7])

        model = RadialBasisFunctionInterpolator(centers, weights, length_scales)
        d = model.to_dict()
        model2 = RadialBasisFunctionInterpolator.from_dict(d)

        x = np.array([[0.1, 0.2], [0.9, 0.8]])
        y1 = model(x)
        y2 = model2(x)
        self.assertTrue(np.allclose(y1, y2))


if __name__ == '__main__':
    unittest.main()



