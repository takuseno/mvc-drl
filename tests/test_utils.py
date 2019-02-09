import numpy as np
import tensorflow as tf

from unittest.mock import MagicMock
from mvc.action_output import ActionOutput
from mvc.models.networks.base_network import BaseNetwork
from mvc.models.metrics.metrics import Metrics


def to_tf(nd_array):
    return tf.constant(nd_array, dtype=tf.float32)


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


def assert_variable_match(variables1, variables2):
    for variable1, variable2 in zip(variables1, variables2):
        assert np.all(variable1 == variable2)


def make_output(num_actions=4, batch_size=1, batch=False):
    if batch:
        action = np.random.random((batch_size, num_actions))
        log_prob = np.random.random((batch_size,))
        value = np.random.random((batch_size,))
    else:
        action = np.random.random((num_actions,))
        log_prob = np.random.random()
        value = np.random.random()
    return ActionOutput(action, log_prob, value)


def make_input(state_size=16, batch_size=1, batch=False):
    if batch:
        obs = np.random.random((batch_size, state_size))
        reward = np.random.random((batch_size,))
        done = np.random.randint(2, size=(batch_size,))
        info = [{'reward': np.random.random()} for _ in range(batch_size)]
    else:
        obs = np.random.random((state_size,))
        reward = np.random.random()
        done = np.random.randint(2)
        info = {'reward': np.random.random()}
    return obs, reward, done, info


class DummyNetwork(BaseNetwork):
    def _infer(self, **kwargs):
        pass

    def _update(self, **kwargs):
        pass

    def _infer_arguments(self):
        pass

    def _update_arguments(self):
        pass


class DummyMetrics(Metrics):
    def __init__(self):
        super().__init__('test')

    def register(self, name, mode, **kwargs):
        pass

    def add(self, name, value):
        pass

    def get(self, name):
        pass

    def log_metric(self):
        pass

    def log_parameters(self):
        pass

    def reset(self):
        pass

    def should_save(self):
        pass

    def save(self):
        pass


class DummyNoise:
    def __init__(self, mock=None):
        self.mock = mock

    def __call__(self):
        if self.mock is not None:
            self.mock()
        return 0.0

    def reset(self):
        pass
