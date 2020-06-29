import unittest

import numpy as np

from tasks import task4 as task4_py


class TestIntegrator(unittest.TestCase):
    def setUp(self):
        # Differential equations with start value
        self.p1 = lambda x: 0.5 * x ** 2 + 2 * x - 3


class TestExplEuler(TestIntegrator):
    def setUp(self):
        super().setUp()

    def test_dims(self):
        x1 = np.zeros((2, 1))
        self.x = task4_py.expl_euler(self.p1, x1, 1)

        self.assertEqual(
            self.x.shape, (2, 1),
            msg="Expl euler result has unexpected size"
        )

    def test_x1(self):
        x1 = np.ones((1, 1))
        self.x1_hist = x1.copy()
        for i in range(3):
            x1 = task4_py.expl_euler(self.p1, x1, 1)
            self.x1_hist = np.append(self.x1_hist, x1)

        x1_expected = np.array([1, 0.5, -1.375, -6.179687])
        self.assertAlmostEqual(
            np.linalg.norm(self.x1_hist - x1_expected), 0, 6,
            msg="Expl euler: Values computed are not as expected."
        )


class TestImplEuler(TestIntegrator):
    def setUp(self):
        super().setUp()

    def test_dims(self):
        x1 = np.zeros((2, 1))
        self.x = task4_py.impl_euler(self.p1, x1, 1)

        self.assertEqual(
            self.x.shape, (2, 1),
            msg="Impl euler result has unexpected size"
        )

    def test_x1(self):
        x1 = np.ones((1, 1))
        self.x1_hist = x1.copy()
        for i in range(3):
            x1 = task4_py.impl_euler(self.p1, x1, 1)
            self.x1_hist = np.append(self.x1_hist, x1)

        x1_expected = np.array([1.0, 1.23606798, 1.12787783, 1.17812863])
        self.assertAlmostEqual(
            np.linalg.norm(self.x1_hist - x1_expected), 0, 6,
            msg="Impl euler: Values computed are not as expected."
        )


class TestHeun(TestIntegrator):
    def setUp(self):
        super().setUp()

    def test_dims(self):
        x1 = np.zeros((2, 1))
        self.x = task4_py.heun(self.p1, x1, 1)

        self.assertEqual(
            self.x.shape, (2, 1),
            msg="Heun result has unexpected size"
        )

    def test_x1(self):
        x1 = np.ones((1, 1))
        self.x1_hist = x1.copy()
        for i in range(3):
            x1 = task4_py.heun(self.p1, x1, 1)
            self.x1_hist = np.append(self.x1_hist, x1)

        x1_expected = np.array([1.0, -0.1875, -3.76951503, -1.21651473])
        self.assertAlmostEqual(
            np.linalg.norm(self.x1_hist - x1_expected), 0, 6,
            msg="Heun: Values computed by are not as expected."
        )


class TestRK(TestIntegrator):
    def setUp(self):
        super().setUp()

    def test_dims(self):
        x1 = np.zeros((2, 1))
        self.x = task4_py.rk4(self.p1, x1, 1)

        self.assertEqual(
            self.x.shape, (2, 1),
            msg="RK result has unexpected size"
        )

    def test_x1(self):
        x1 = np.ones((1, 1))
        self.x1_hist = x1.copy()
        for i in range(3):
            x1 = task4_py.rk4(self.p1, x1, 1)
            self.x1_hist = np.append(self.x1_hist, x1)

        x1_expected = np.array([1.0, -0.97578688, -4.49632406, -3.86973236])
        self.assertAlmostEqual(
            np.linalg.norm(self.x1_hist - x1_expected), 0, 6,
            msg="RK: Values computed by are not as expected."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
