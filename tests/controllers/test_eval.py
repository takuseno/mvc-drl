import numpy as np
import pytest

from unittest import TestCase
from unittest.mock import MagicMock
from mvc.action_output import ActionOutput
from mvc.controllers.eval import EvalController
from mvc.models.metrics import Metrics
from mvc.models.networks.base_network import BaseNetwork


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

def make_inputs():
    obs = np.random.random((4, 84, 84))
    reward = np.random.random((4,))
    info = [{} for _ in range(4)]
    done = np.array([np.random.randint(2) for _ in range(4)])
    return [obs, reward, done, info]

def make_output():
    action = np.random.random((4, np.random.randint(10) + 1))
    log_prob = np.random.random((4, action.shape[1]))
    value = np.random.random((4,))
    return ActionOutput(action, log_prob, value)

class EvalControllerTest(TestCase):
    def test_step(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        output = make_output()
        network.infer = MagicMock(return_value=output)
        metrics.get = MagicMock(return_value=0)

        inputs = make_inputs()
        inputs[2] = np.zeros((4,))
        controller.step(*inputs)

        network.infer.assert_called_once()
        assert metrics.get.call_count == 4

    def test_step_with_done(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        output = make_output()
        network.infer = MagicMock(return_value=output)
        metrics.add = MagicMock()
        metrics.get = MagicMock(return_value=1)

        inputs = make_inputs()
        index = np.random.randint(4)
        reward = np.random.random()
        inputs[2] = np.zeros((4,))
        inputs[2][index] = 1.0
        inputs[3][index]['reward'] = reward
        controller.step(*inputs)

        assert metrics.add.call_count == 2
        assert list(metrics.add.mock_calls[1])[1] == ('eval_reward', reward)
        assert list(metrics.add.mock_calls[0])[1] == ('eval_episode', 1)

    def test_step_with_eval_episode_over_limit(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        output = make_output()
        network.infer = MagicMock(return_value=output)
        metrics.add = MagicMock(side_effect=Exception)
        metrics.get = MagicMock(return_value=10)

        inputs = make_inputs()
        index = np.random.randint(4)
        reward = np.random.random()
        inputs[2] = np.zeros((4,))
        inputs[2][index] = 1.0
        inputs[3][index]['reward'] = reward
        controller.step(*inputs)

    def test_stop_episode(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        metrics.add = MagicMock()

        obs = np.random.random((84, 84, 4))
        reward = np.random.random()
        info = {'reward': np.random.random()}
        controller.stop_episode(obs, reward, info)

        assert metrics.add.call_count == 2
        assert list(metrics.add.mock_calls[0])[1] == ('eval_reward', info['reward'])
        assert list(metrics.add.mock_calls[1])[1] == ('eval_episode', 1)

    def test_should_update(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        assert not controller.should_update()

    def test_update(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        with pytest.raises(Exception):
            controller.update()

    def test_should_log(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        metrics.get = MagicMock(return_value=9)
        assert not controller.should_log()
        metrics.get.assert_called_once_with('eval_episode')

        metrics.get = MagicMock(return_value=10)
        assert controller.should_log()
        metrics.get.assert_called_once_with('eval_episode')

    def test_log(self):
        network = DummyNetwork()
        metrics = DummyMetrics()
        metrics.has = MagicMock(return_value=True)
        controller = EvalController(network, metrics, 10)

        metrics.get = MagicMock(return_value=5)
        metrics.log_metric = MagicMock()
        metrics.reset = MagicMock()

        controller.log()

        metrics.get.assert_called_once_with('step')
        metrics.log_metric.assert_called_once_with('eval_reward', 5)
        assert list(metrics.reset.mock_calls[0])[1] == ('eval_episode',)
        assert list(metrics.reset.mock_calls[1])[1] == ('eval_reward',)
