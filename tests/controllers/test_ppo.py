import numpy as np
import unittest
import pytest
import copy
import mvc.logger as logger

from unittest.mock import MagicMock, Mock
from mvc.models.networks.base_network import BaseNetwork
from mvc.controllers.ppo import PPOController
from mvc.models.rollout import Rollout
from mvc.models.metrics import Metrics
from mvc.preprocess import compute_returns, compute_gae
from mvc.action_output import ActionOutput


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

def make_output():
    action = np.random.random((4, 4))
    log_prob = np.random.random((4, 4))
    value = np.random.random((4,))
    return ActionOutput(action, log_prob, value)

def make_inputs():
    obs = np.random.random((4, 84, 84))
    reward = np.random.random((4,))
    done = np.zeros((4,))
    info = [{'reward': np.random.random()} for _ in range(4)]
    return obs, reward, done, info

class PPOControllerTest(unittest.TestCase):
    def test_init_success(self):
        network = DummyNetwork()
        rollout = Rollout()
        metrics = DummyMetrics()
        controller = PPOController(network, rollout, metrics, num_envs=4,
                                   time_horizon=128, epoch=4, batch_size=32,
                                   gamma=0.99, lam=0.9)

    def test_init_with_error(self):
        with pytest.raises(AssertionError):
            network = 'network'
            rollout = Rollout()
            metrics = DummyMetrics()
            controller = PPOController(network, rollout, metrics, num_envs=4,
                                       time_horizon=128, epoch=4, batch_size=32,
                                       gamma=0.99, lam=0.9)
        with pytest.raises(AssertionError):
            network = DummyNetwork()
            rollout = 'rollout'
            metrics = DummyMetrics()
            controller = PPOController(network, rollout, metrics, num_envs=4,
                                       time_horizon=128, epoch=4, batch_size=32,
                                       gamma=0.99, lam=0.9)
        with pytest.raises(AssertionError):
            network = DummyNetwork()
            rollout = Rollout()
            metrics = 'metrics'
            controller = PPOController(network, rollout, metrics, num_envs=4,
                                       time_horizon=128, epoch=4, batch_size=32,
                                       gamma=0.99, lam=0.9)

    def test_step(self):
        rollout = Rollout()
        network = DummyNetwork()
        metrics = DummyMetrics()
        output = make_output()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, metrics, num_envs=4,
                                   time_horizon=128, epoch=4, batch_size=32,
                                   gamma=0.99, lam=0.9)
        inputs = make_inputs()
        action = controller.step(*inputs)

        assert np.all(action == output.action)
        self.assertEqual(rollout.size(), 1)
        assert np.all(inputs[0] == rollout.obs_t[0])
        assert np.all(inputs[1] == rollout.rewards_t[0])
        assert np.all(inputs[2] == rollout.terminals_t[0])
        assert np.all(output.action == rollout.actions_t[0])
        assert np.all(output.value == rollout.values_t[0])
        assert np.all(output.log_prob == rollout.log_probs_t[0])

    def test_should_update(self):
        rollout = Rollout()
        network = DummyNetwork()
        metrics = DummyMetrics()
        output = make_output()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, metrics, num_envs=4,
                                   time_horizon=128, epoch=4, batch_size=32,
                                   gamma=0.99, lam=0.9)
        inputs = make_inputs()
        for i in range(128):
            controller.step(*inputs)
            self.assertFalse(controller.should_update())
        controller.step(*inputs)
        self.assertTrue(controller.should_update())

    def test_batches(self):
        rollout = Rollout()
        network = DummyNetwork()
        metrics = DummyMetrics()
        output = make_output()
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, metrics, num_envs=4,
                                   time_horizon=128, epoch=4, batch_size=32,
                                   gamma=0.99, lam=0.9)
        input_history = []
        output_history = []
        for i in range(129):
            inputs = make_inputs()
            network._infer = MagicMock(return_value=output)
            action = controller.step(*inputs)
            input_history.append(inputs)
            output_history.append(output)

        batches = controller._batches()

        assert len(batches) == 128 * 4 // 32
        for key in ['obs_t', 'actions_t', 'log_probs_t', 'returns_t', 'advantages_t']:
            for batch in batches:
                assert key in batch
                assert batch[key].shape[0] == 32
                if key == 'obs_t':
                    assert batch[key].shape[1:] == inputs[0].shape[1:]
                elif key == 'actions_t':
                    assert batch[key].shape[1] == action.shape[1]
                elif key == 'log_probs_t':
                    assert batch[key].shape[1] == action.shape[1]
                elif key == 'returns_t':
                    assert len(batch[key].shape) == 1
                elif key == 'advantages_t':
                    assert len(batch[key].shape) == 1

    def test_batch_with_short_trajectory_error(self):
        rollout = Rollout()
        network = DummyNetwork()
        metrics = DummyMetrics()
        output = make_output()
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        network._infer = MagicMock(return_value=output)
        controller = PPOController(network, rollout, metrics, num_envs=4,
                                   time_horizon=128, epoch=4, batch_size=32,
                                   gamma=0.99, lam=0.9)
        inputs = make_inputs()
        action = controller.step(*inputs)
        with pytest.raises(AssertionError):
            controller._batches()

    def test_update_with_should_update_false(self):
        rollout = Rollout()
        network = DummyNetwork()
        metrics = DummyMetrics()
        inputs = make_inputs()
        output = make_output()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, metrics, num_envs=4,
                                   time_horizon=128, epoch=4, batch_size=32,
                                   gamma=0.99, lam=0.9)
        for i in range(20):
            action = controller.step(*inputs)
        with pytest.raises(AssertionError):
            controller.update()

    def test_update_success(self):
        rollout = Rollout()
        network = DummyNetwork()
        metrics = DummyMetrics()
        inputs = make_inputs()
        output = make_output()
        loss = np.random.random()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        network._update_arguments = MagicMock(return_value=['obs_t', 'actions_t', 'returns_t', 'advantages_t', 'log_probs_t'])
        network._update = MagicMock(return_value=loss)
        controller = PPOController(network, rollout, metrics, num_envs=4,
                                   time_horizon=128, epoch=4, batch_size=32,
                                   gamma=0.99, lam=0.9)
        for i in range(129):
            action = controller.step(*inputs)

        assert np.allclose(controller.update(), loss)
        assert rollout.size() == 0
        assert network._update.call_count == 128 * 4 * 4 // 32
