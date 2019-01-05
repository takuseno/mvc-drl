import numpy as np
import pytest
import unittest

from mvc.preprocess import compute_returns
from mvc.preprocess import compute_gae


class ComputeReturnsTest(unittest.TestCase):
    def test_compute_returns_one_d_array(self):
        bootstrap_value = 1.0
        rewards = np.array([1.0, 2.0, 3.0])
        terminals = np.array([0.0, 1.0, 0.0])
        answer = np.array([2.8, 2.0, 3.9])
        returns = compute_returns(bootstrap_value, rewards, terminals, 0.9)
        self.assertTrue(np.all(returns == answer))

    def test_compute_returns_two_d_array(self):
        bootstrap_value = np.array([1.0, 2.0, 0.0])
        rewards = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        terminals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        answer = np.array([[3.61, 4.7, 3.0], [2.9, 3.0, 4.0]])
        returns = compute_returns(bootstrap_value, rewards, terminals, 0.9)
        self.assertTrue(np.all(returns == answer))

    def test_compute_returns_not_ndarray_error(self):
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(0.0, np.arange(5), range(5), 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(0.0, range(5), np.arange(5), 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(0.0, np.ones((5, 5)), np.ones((5, 5)), 0.9)

    def test_compute_returns_shape_does_not_match_error(self):
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(0.0, range(5), np.arange(6), 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(0.0, range(6), np.arange(5), 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(np.arange(5), np.ones((6, 5)), np.ones((5, 5)), 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(np.arange(5), np.ones((5, 5)), np.ones((6, 5)), 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_returns(np.arange(6), np.ones((5, 5)), np.ones((5, 5)), 0.9)

class ComputeGaeTest(unittest.TestCase):
    def test_compute_returns_one_d_array(self):
        bootstrap_value = 1.0
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([2.0, 4.0, 6.0])
        terminals = np.array([0.0, 1.0, 0.0])
        answer = np.array([0.98, -2.0, -2.1])
        advs = compute_gae(bootstrap_value, rewards, values, terminals, 0.9, 0.9)
        self.assertTrue(np.allclose(advs, answer))

    def test_compute_returns_two_d_array(self):
        bootstrap_value = np.array([1.0, 2.0, 0.0])
        rewards = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        values = np.array([[2.0, 4.0, 6.0], [3.0, 0.0, 5.0]])
        terminals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        answer = np.array([[1.619, 0.43, -3.0], [-0.1, 3.0, -1.0]])
        advs = compute_gae(bootstrap_value, rewards, values, terminals, 0.9, 0.9)
        self.assertTrue(np.allclose(advs, answer))

    def test_compute_returns_not_ndarray_error(self):
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(0.0, np.arange(5), np.arange(5), range(5), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(0.0, np.arange(5), range(5), np.arange(5), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(0.0, range(5), np.arange(5), np.arange(5), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(0.0, np.ones((5, 5)), np.ones((5, 5)), np.ones((5, 5)), 0.9, 0.9)

    def test_compute_returns_shape_does_not_match_error(self):
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(0.0, np.arange(5), np.arange(5), np.arange(6), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(0.0, np.arange(5), np.arange(6), np.arange(5), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(0.0, np.arange(6), np.arange(5), np.arange(5), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(np.arange(5), np.ones((6, 5)), np.ones((5, 5)), np.ones((5, 5)), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(np.arange(5), np.ones((5, 5)), np.ones((6, 5)), np.ones((5, 5)), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(np.arange(5), np.ones((5, 5)), np.ones((5, 5)), np.ones((6, 5)), 0.9, 0.9)
        with pytest.raises(AssertionError) as excinfo:
            compute_gae(np.arange(6), np.ones((5, 5)), np.ones((5, 5)), np.ones((5, 5)), 0.9, 0.9)
