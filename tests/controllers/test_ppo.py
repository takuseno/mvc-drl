import numpy as np
import unittest
import pytest
import copy

from unittest.mock import MagicMock, Mock
from mvc.controllers.ppo import PPOController
from mvc.models.rollout import Rollout
from mvc.preprocess import compute_returns, compute_gae
from tests.test_utils import make_input, make_output
from tests.test_utils import DummyNetwork, DummyMetrics


class TestPPOController:
    def setup_method(self):
        self.network = DummyNetwork()
        self.rollout = Rollout()
        self.metrics = DummyMetrics()
        self.controller = PPOController(
            self.network, self.rollout, self.metrics, num_envs=4,
            time_horizon=128, epoch=4, batch_size=32, gamma=0.99, lam=0.9)

    def test_step(self):
        output = make_output(batch_size=4, batch=True)
        self.network._infer = MagicMock(return_value=output)
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])

        inpt = make_input(batch_size=4, batch=True)
        action = self.controller.step(*inpt)

        assert np.all(action == output.action)
        assert self.rollout.size() == 1
        assert np.all(inpt[0] == self.rollout.obs_t[0])
        assert np.all(inpt[1] == self.rollout.rewards_t[0])
        assert np.all(inpt[2] == self.rollout.terminals_t[0])
        assert np.all(output.action == self.rollout.actions_t[0])
        assert np.all(output.value == self.rollout.values_t[0])
        assert np.all(output.log_prob == self.rollout.log_probs_t[0])

    def test_should_update(self):
        output = make_output(batch_size=4, batch=True)
        self.network._infer = MagicMock(return_value=output)
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])

        inpt = make_input(batch_size=4, batch=True)
        for i in range(128):
            self.controller.step(*inpt)
            assert not self.controller.should_update()
        self.controller.step(*inpt)
        assert self.controller.should_update()

    def test_batches(self):
        output = make_output(batch_size=4, batch=True)
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])

        input_history = []
        output_history = []
        for i in range(129):
            inpt = make_input(batch_size=4, batch=True)
            self.network._infer = MagicMock(return_value=output)
            action = self.controller.step(*inpt)
            input_history.append(inpt)
            output_history.append(output)

        for key in ['obs_t', 'actions_t', 'log_probs_t', 'returns_t', 'advantages_t', 'values_t']:
            count = 0
            for batch in self.controller._batches():
                count += 1
                assert key in batch
                assert batch[key].shape[0] == 32
                if key == 'obs_t':
                    assert batch[key].shape[1:] == inpt[0].shape[1:]
                elif key == 'actions_t':
                    assert batch[key].shape[1] == action.shape[1]
                elif key == 'log_probs_t':
                    assert len(batch[key].shape) == 1
                elif key == 'returns_t':
                    assert len(batch[key].shape) == 1
                elif key == 'advantages_t':
                    assert len(batch[key].shape) == 1
                elif key == 'values_t':
                    assert len(batch[key].shape) == 1
            assert count == 128 * 4 // 32

    def test_batch_with_short_trajectory_error(self):
        output = make_output(batch_size=4, batch=True)
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])
        self.network._infer = MagicMock(return_value=output)

        inpt = make_input(batch_size=4, batch=True)
        action = self.controller.step(*inpt)
        with pytest.raises(AssertionError):
            self.controller._batches()

    def test_update_with_should_update_false(self):
        inpt = make_input(batch_size=4, batch=True)
        output = make_output(batch_size=4, batch=True)
        self.network._infer = MagicMock(return_value=output)
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])

        for i in range(20):
            action = self.controller.step(*inpt)
        with pytest.raises(AssertionError):
            self.controller.update()

    def test_update_success(self):
        inpt = make_input(batch_size=4, batch=True)
        output = make_output(batch_size=4, batch=True)
        loss = np.random.random()
        self.network._infer = MagicMock(return_value=output)
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])
        self.network._update_arguments = MagicMock(return_value=['obs_t', 'actions_t', 'returns_t', 'advantages_t', 'log_probs_t'])
        self.network._update = MagicMock(return_value=loss)

        for i in range(129):
            action = self.controller.step(*inpt)

        assert np.allclose(self.controller.update(), loss)
        assert self.rollout.size() == 0
        assert self.network._update.call_count == 128 * 4 * 4 // 32

    def test_log(self):
        step = np.random.randint(100)
        self.metrics.get = MagicMock(return_value=step)
        self.metrics.log_metric = MagicMock()

        self.controller.log()

        self.metrics.get.assert_called_once_with('step')
        assert tuple(self.metrics.log_metric.call_args_list[0])[0] == ('reward', step)
        assert tuple(self.metrics.log_metric.call_args_list[1])[0] == ('loss', step)
