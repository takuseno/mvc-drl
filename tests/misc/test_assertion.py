import numpy as np
import pytest

from mvc.misc.assertion import assert_type
from mvc.misc.assertion import assert_shape
from mvc.misc.assertion import assert_batch_shape
from mvc.misc.assertion import assert_batch_size_match
from mvc.misc.assertion import assert_shape_match
from mvc.misc.assertion import assert_shape_length
from mvc.misc.assertion import assert_scalar

def test_assert_type():
    with pytest.raises(AssertionError):
        assert_type('error', int)

    assert_type('error', str)


def test_assert_shape():
    with pytest.raises(AssertionError):
        assert_shape(np.random.random((4, 10)), (4,))

    assert_shape(np.random.random((4, 10)), (4, 10))


def test_assert_batch_shape():
    with pytest.raises(AssertionError):
        assert_batch_shape(np.random.random((4, 10)), (11,))

    assert_batch_shape(np.random.random((4, 10)), (10,))


def test_assert_batch_size_match():
    with pytest.raises(AssertionError):
        assert_batch_size_match(np.random.random((4, 10)), np.random.random((5, 10)))

    assert_batch_size_match(np.random.random((4, 10)), np.random.random((4, 9)))


def test_assert_shape_match():
    with pytest.raises(AssertionError):
        assert_shape_match(np.random.random((4, 10)), np.random.random((4, 11)))

    assert_shape_match(np.random.random((4, 10)), np.random.random((4, 10)))


def test_assert_shape_length():
    with pytest.raises(AssertionError):
        assert_shape_length(np.random.random((4, 10)), 3)

    assert_shape_length(np.random.random((4, 10)), 2)


def test_assert_scalar():
    with pytest.raises(AssertionError):
        assert_scalar(np.random.random((4,)))
    with pytest.raises(AssertionError):
        assert_scalar(np.random.random((4, 10)))

    assert_scalar(np.random.random((4, 1)))
