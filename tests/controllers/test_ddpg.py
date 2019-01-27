import numpy as np
import unittest
import pytest

from unittest.mock import MagicMock
from mvc.models.networks.base_network import BaseNetwork
from mvc.controllers.ddpg import DDPGController
from mvc.models.buffer import Buffer
from mvc.models.metrics import Metrics
from mvc.action_output import ActionOutput
from mvc.noise import OrnsteinUhlenbeckActionNoise


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

def make_output():
    action = np.random.random((4,))
    log_prob = None
    value = np.random.random()
    return ActionOutput(action, log_prob, value)

def make_inputs():
    obs = np.random.random((16,))
    reward = np.random.random()
    done = 0.0
    info = {'reward': np.random.random()}
    return obs, reward, done, info


class DDPGControllerTest(unittest.TestCase):
    def test_init(self):
        network = DummyNetwork()
        buffer = Buffer()
        metrics = DummyMetrics()
        noise = DummyNoise()
        controller = DDPGController(network, buffer, metrics, noise,
                                    num_actions=4, batch_size=32)

    def test_step(self):
        network = DummyNetwork()
        buffer = Buffer()
        metrics = DummyMetrics()
        noise = DummyNoise(MagicMock())
        output = make_output()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        metrics.add = MagicMock()
        controller = DDPGController(network, buffer, metrics, noise,
                                    num_actions=4, batch_size=32)

        inputs = make_inputs()
        action = controller.step(*inputs)

        assert buffer.size() == 1
        assert np.all(output.action == action)
        metrics.add.assert_called_once_with('step', 1)
        noise.mock.assert_called_once()

        action = controller.step(*inputs)
        assert buffer.size() == 2

    def test_should_update(self):
        network = DummyNetwork()
        buffer = Buffer()
        metrics = DummyMetrics()
        noise = DummyNoise()
        controller = DDPGController(network, buffer, metrics, noise,
                                    num_actions=4, batch_size=32)

        buffer.size = MagicMock(return_value=np.random.randint(32))
        assert not controller.should_update()

        buffer.size = MagicMock(return_value=33)
        assert controller.should_update()

    def test_update(self):
        network = DummyNetwork()
        buffer = Buffer()
        metrics = DummyMetrics()
        noise = DummyNoise()
        critic_loss = np.random.random()
        actor_loss = np.random.random()
        network._update = MagicMock(return_value=(critic_loss, actor_loss))
        network._update_arguments = MagicMock(
            return_value=['obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1'])
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        network._infer = MagicMock(return_value=ActionOutput(np.zeros(1), None, np.zeros(1)))
        metrics.add = MagicMock()
        controller = DDPGController(network, buffer, metrics, noise,
                                    num_actions=4, batch_size=32)

        for i in range(33):
            inputs = make_inputs()
            controller.step(*inputs)
        controller.update()

        network._update.assert_called_once()

    def test_log(self):
        network = DummyNetwork()
        buffer = Buffer()
        metrics = DummyMetrics()
        noise = DummyNoise()
        step = np.random.randint(10) + 1
        metrics.get = MagicMock(return_value=step)
        metrics.log_metric = MagicMock()
        controller = DDPGController(network, buffer, metrics, noise,
                                    num_actions=4, batch_size=32)

        controller.log()

        metrics.get.assert_called_once_with('step')
        assert metrics.log_metric.call_count == 3
        assert metrics.log_metric.call_args[0] == ('actor_loss', step)

    def stop_episode(self):
        network = DummyNetwork()
        buffer = Buffer()
        metrics = DummyMetrics()
        noise = DummyNoise()
        buffer.add = MagicMock()
        metrics.add = MagicMock()
        noise.reset = MagicMock()
        controller = DDPGController(network, buffer, metrics, noise,
                                    num_actions=4, batch_size=32)

        inputs = make_inputs()
        controller.stop_episode(inputs[0], inputs[1], inputs[3])

        assert np.all(buffer.add.call_args[0][3] == inputs[0])
        assert np.all(buffer.add.call_args[0][2] == np.zeros())
        assert buffer.add.call_args[0][1] == inputs[1]
        assert buffer.add.call_args[0][0] == 1.0
        metrics.add.assert_called_once_with('reward', inputs[3]['reward'])
        noise.reset.assert_called_once()
