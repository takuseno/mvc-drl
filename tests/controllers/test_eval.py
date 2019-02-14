import numpy as np
import pytest

from unittest import TestCase
from unittest.mock import MagicMock
from mvc.controllers.eval import EvalController
from tests.test_utils import DummyNetwork, DummyMetrics
from tests.test_utils import make_input, make_output


class TestEvalController:
    def setup_method(self):
        self.network = DummyNetwork()
        self.metrics = DummyMetrics()
        self.metrics.has = MagicMock(return_value=True)
        self.controller = EvalController(self.network, self.metrics, 10)

    @pytest.mark.parametrize("batch", [True, False])
    def test_step(self, batch):
        output = make_output()
        self.network.infer = MagicMock(return_value=output)
        self.metrics.get = MagicMock(return_value=0)

        if batch:
            inpt = list(make_input(batch_size=4, batch=True))
            inpt[2] = np.zeros((4,))
        else:
            inpt = list(make_input())
            inpt[2] = 0.0
        step_output = self.controller.step(*inpt)

        assert step_output is output.action
        assert self.network.infer.call_count == 1
        if batch:
            assert self.metrics.get.call_count == 4
        else:
            assert self.metrics.get.call_count == 0

    @pytest.mark.parametrize("batch", [True, False])
    def test_step_with_done(self, batch):
        output = make_output()
        self.network.infer = MagicMock(return_value=output)
        self.metrics.add = MagicMock()
        self.metrics.get = MagicMock(return_value=1)

        reward = np.random.random()
        if batch:
            inpt = list(make_input(batch_size=4, batch=True))
            index = np.random.randint(4)
            inpt[2] = np.zeros((4,))
            inpt[2][index] = 1.0
            inpt[3][index]['reward'] = reward
        else:
            inpt = list(make_input())
            inpt[2] = 1.0
            inpt[3]['reward'] = reward
        self.controller.step(*inpt)

        if batch:
            assert self.metrics.add.call_count == 2
            assert list(self.metrics.add.mock_calls[1])[1] == ('eval_reward', reward)
            assert list(self.metrics.add.mock_calls[0])[1] == ('eval_episode', 1)
        else:
            self.metrics.add.assert_not_called()

    def test_step_with_eval_episode_over_limit(self):
        output = make_output()
        self.network.infer = MagicMock(return_value=output)
        self.metrics.add = MagicMock(side_effect=Exception)
        self.metrics.get = MagicMock(return_value=10)

        inpt = list(make_input(batch_size=4, batch=True))
        index = np.random.randint(4)
        reward = np.random.random()
        inpt[2] = np.zeros((4,))
        inpt[2][index] = 1.0
        inpt[3][index]['reward'] = reward
        self.controller.step(*inpt)

    def test_stop_episode(self):
        self.metrics.add = MagicMock()

        obs, reward, _, info = make_input()
        self.controller.stop_episode(obs, reward, info)

        assert self.metrics.add.call_count == 2
        assert list(self.metrics.add.mock_calls[0])[1] == ('eval_reward', info['reward'])
        assert list(self.metrics.add.mock_calls[1])[1] == ('eval_episode', 1)

    def test_should_update(self):
        assert not self.controller.should_update()

    def test_update(self):
        with pytest.raises(Exception):
            self.controller.update()

    def test_should_log(self):
        assert not self.controller.should_log()

    def test_log(self):
        with pytest.raises(Exception):
            self.controller.log()

    def test_is_finished(self):
        self.metrics.get = MagicMock(return_value=5)
        self.metrics.reset = MagicMock()
        self.metrics.log_metric = MagicMock()

        assert not self.controller.is_finished()
        self.metrics.reset.assert_not_called()
        self.metrics.log_metric.assert_not_called()

        self.metrics.get = MagicMock(return_value=10)
        assert self.controller.is_finished()
        assert list(self.metrics.reset.mock_calls[0])[1] == ('eval_episode',)
        assert list(self.metrics.reset.mock_calls[1])[1] == ('eval_reward',)
        self.metrics.log_metric.assert_called_once_with('eval_reward', 10)

    def test_should_save(self):
        assert not self.controller.should_save()

    def test_save(self):
        self.metrics.save_model = MagicMock()
        self.controller.save()
        self.metrics.save_model.assert_not_called()

    def test_should_eval(self):
        assert not self.controller.should_eval()
