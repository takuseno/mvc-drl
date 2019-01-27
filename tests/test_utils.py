import numpy as np
import tensorflow as tf

from unittest.mock import MagicMock


def make_tf_inpt():
    dim1 = np.random.randint(10) + 1
    dim2 = np.random.randint(10) + 1
    return tf.constant(np.random.random((dim1, dim2)), dtype=tf.float32)


def make_fcs():
    return [np.random.randint(100) + 1 for i in range(np.random.randint(5) + 1)]


def mock_activation():
    return MagicMock(side_effect=tf.nn.relu)


def assert_hidden_variable_shape(variables, inpt, fcs):
    shapes = [(int(inpt.shape[1]), fcs[0])]
    shapes += shapes
    for i, fc in enumerate(fcs[1:]):
        shapes += [(fcs[i], fc)]
        shapes += [(fcs[i], fc)]
    for shape, variable in zip(shapes, variables):
        if str(variable.name).find('kernel') > 0:
            assert (int(variable.shape[0]), int(variable.shape[1])) == shape
        else:
            assert int(variable.shape[0]) == shape[1]


def assert_variable_range(variable, min_val, max_val):
    assert not np.any(variable < min_val)
    assert not np.any(variable > max_val)


def assert_variable_mismatch(variables1, variables2):
    for variable1, variable2 in zip(variables1, variables2):
        assert not np.all(variable1 == variable2)
