import numpy as np
import unittest
import pytest

from unittest.mock import MagicMock, Mock
from mvc.models.networks.base_network import BaseNetwork
from mvc.controllers.ppo import PPOController
from mvc.models.rollout import Rollout
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

def make_output():
    action = np.random.random((4, 4))
    log_prob = np.random.random((4,))
    value = np.random.random((4,))
    return ActionOutput(action, log_prob, value)

def make_inputs():
    obs = np.random.random((4, 84, 84))
    reward = np.random.random((4,))
    done = np.zeros((4,))
    return obs, reward, done

class PPOControllerTest(unittest.TestCase):
    def test_init_success(self):
        network = DummyNetwork()
        rollout = Rollout()
        controller = PPOController(network, rollout, 20, 0.99, 0.9)

    def test_init_with_error(self):
        with pytest.raises(AssertionError):
            network = 'network'
            rollout = Rollout()
            cotroller = PPOController(network, rollout, 20, 0.99, 0.9)
        with pytest.raises(AssertionError):
            network = DummyNetwork()
            rollout = 'rollout'
            cotroller = PPOController(network, rollout, 20, 0.99, 0.9)

    def test_step(self):
        rollout = Rollout()
        network = DummyNetwork()
        output = make_output()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, 20, 0.99, 0.9)
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
        output = make_output()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, 20, 0.99, 0.9)
        inputs = make_inputs()
        for i in range(20):
            controller.step(*inputs)
            self.assertFalse(controller.should_update())
        controller.step(*inputs)
        self.assertTrue(controller.should_update())

    def test_batch_success(self):
        rollout = Rollout()
        network = DummyNetwork()
        output = make_output()
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, 20, 0.99, 0.9)
        input_history = []
        output_history = []
        for i in range(21):
            inputs = make_inputs()
            network._infer = MagicMock(return_value=output)
            action = controller.step(*inputs)
            input_history.append(inputs)
            output_history.append(output)

        batch = controller._batch()
        for key in ['obs_t', 'actions_t', 'log_probs_t', 'returns_t', 'advantages_t']:
            assert key in batch

        obs_t = np.array([inputs[0] for inputs in input_history])
        assert np.all(batch['obs_t'] == obs_t[:-1])

        actions_t = np.array([output.action for output in output_history])
        assert np.all(batch['actions_t'] == actions_t[:-1])

        log_probs_t = np.array([output.log_prob for output in output_history])
        assert np.all(batch['log_probs_t'] == log_probs_t[:-1])

        rewards_tp1 = np.array([inputs[1] for inputs in input_history])[1:]
        terminals_tp1 = np.array([inputs[2] for inputs in input_history])[1:]
        values_t = np.array([output.value for output in output_history])[:-1]
        bootstrap_value = np.array(output_history[-1].value)
        returns_t = compute_returns(bootstrap_value, rewards_tp1, terminals_tp1, 0.99)
        assert np.all(returns_t == batch['returns_t'])
        advantages_t = compute_gae(bootstrap_value, rewards_tp1, values_t,
                                   terminals_tp1, 0.99, 0.9)
        assert np.all(advantages_t == batch['advantages_t'])

    def test_update_with_should_update_false(self):
        rollout = Rollout()
        network = DummyNetwork()
        inputs = make_inputs()
        output = make_output()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        controller = PPOController(network, rollout, 20, 0.99, 0.9)
        for i in range(20):
            action = controller.step(*inputs)
        with pytest.raises(AssertionError):
            controller.update()

    def test_update_success(self):
        rollout = Rollout()
        network = DummyNetwork()
        inputs = make_inputs()
        output = make_output()
        loss = np.random.random()
        network._infer = MagicMock(return_value=output)
        network._infer_arguments = MagicMock(return_value=['obs_t'])
        network._update_arguments = MagicMock(return_value=['obs_t', 'actions_t', 'returns_t', 'advantages_t', 'log_probs_t'])
        network._update = MagicMock(return_value=loss)
        controller = PPOController(network, rollout, 20, 0.99, 0.9)
        for i in range(21):
            action = controller.step(*inputs)
        
        self.assertEqual(controller.update(), loss)
        self.assertEqual(rollout.size(), 0)
        network._update.assert_called_once()
