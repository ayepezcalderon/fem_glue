import functools
import unittest

from fem_glue._utils import tol_compare


class TestTolCompare(unittest.TestCase):
    tol_compare = functools.partial(tol_compare, tol=0.05)

    def test_lt(self):
        self.assertTrue(self.tol_compare(0.94, 1.0, "lt"))
        self.assertFalse(self.tol_compare(0.97, 1.0, "lt"))

    def test_le(self):
        self.assertTrue(self.tol_compare(1.0, 1.07, "le"))
        self.assertTrue(self.tol_compare(1.0, 0.97, "le"))
        self.assertFalse(self.tol_compare(1.0, 0.94, "le"))

    def test_eq(self):
        self.assertTrue(self.tol_compare(1.04, 1.0, "eq"))
        self.assertTrue(self.tol_compare(0.96, 1.0, "eq"))
        self.assertFalse(self.tol_compare(0.94, 1.0, "eq"))

    def test_ne(self):
        self.assertFalse(self.tol_compare(1.04, 1.0, "ne"))
        self.assertFalse(self.tol_compare(0.96, 1.0, "ne"))
        self.assertTrue(self.tol_compare(0.94, 1.0, "ne"))

    def test_ge(self):
        self.assertTrue(self.tol_compare(1.07, 1.0, "ge"))
        self.assertTrue(self.tol_compare(0.97, 1.0, "ge"))
        self.assertFalse(self.tol_compare(0.94, 1.0, "ge"))

    def test_gt(self):
        self.assertTrue(self.tol_compare(1.07, 1.0, "gt"))
        self.assertFalse(self.tol_compare(1.04, 1.0, "gt"))

    def test_invalid_op(self):
        with self.assertRaises(ValueError):
            self.tol_compare(1.0, 1.02, "invalid_op")  # type: ignore
