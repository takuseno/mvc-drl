import numpy as np
import unittest
import pytest

from unittest.mock import MagicMock
from mvc.models.networks.base_network import BaseNetwork
from mvc.controllers.ddpg import DDPGController
from mvc.models.buffer import Buffer
from mvc.models.metrics import Metrics
from mvc.noise import OrnsteinUhlenbeckActionNoise
from tests.test_utils import make_output, make_input
from tests.test_utils import DummyNetwork, DummyMetrics, DummyNoise


class DDPGControllerTest(unittest.TestCase):
    def setUp(self):
        self.network = DummyNetwork()
        self.buffer = Buffer()
        self.metrics = DummyMetrics()
        self.noise = DummyNoise()
        self.controller = DDPGController(self.network, self.buffer,
                                         self.metrics, self.noise,
                                         num_actions=4, batch_size=32)

    def test_register_metrics(self):
        self.metrics.register = MagicMock()
        self.controller._register_metrics()

        assert tuple(self.metrics.register.call_args_list[0])[0] == ('step', 'single')
        assert tuple(self.metrics.register.call_args_list[1])[0] == ('critic_loss', 'queue')
        assert tuple(self.metrics.register.call_args_list[2])[0] == ('actor_loss', 'queue')
        assert tuple(self.metrics.register.call_args_list[3])[0] == ('reward', 'queue')

    def test_record_update_metrics(self):
        self.metrics.add = MagicMock()

        critic_loss = np.random.random()
        actor_loss = np.random.random()
        self.controller._record_update_metrics(critic_loss, actor_loss)

        assert tuple(self.metrics.add.call_args_list[0])[0] == ('critic_loss', critic_loss)
        assert tuple(self.metrics.add.call_args_list[1])[0] == ('actor_loss', actor_loss)

    def test_step(self):
        output = make_output()
        self.noise.mock = MagicMock()
        self.network._infer = MagicMock(return_value=output)
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])
        self.metrics.add = MagicMock()

        inpt = make_input()
        action = self.controller.step(*inpt)

        assert self.buffer.size() == 1
        assert np.all(output.action == action)
        self.metrics.add.assert_called_once_with('step', 1)
        assert self.noise.mock.call_count == 1

        action = self.controller.step(*inpt)
        assert self.buffer.size() == 2

    def test_should_update(self):
        self.buffer.size = MagicMock(return_value=np.random.randint(32))
        assert not self.controller.should_update()

        self.buffer.size = MagicMock(return_value=33)
        assert self.controller.should_update()

    def test_update(self):
        critic_loss = np.random.random()
        actor_loss = np.random.random()
        output = make_output()
        self.network._update = MagicMock(return_value=(critic_loss, actor_loss))
        self.network._update_arguments = MagicMock(return_value=['obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1'])
        self.network._infer_arguments = MagicMock(return_value=['obs_t'])
        self.network._infer = MagicMock(return_value=output)
        self.metrics.add = MagicMock()
        self.controller._record_update_metrics = MagicMock()

        for i in range(33):
            inpt = make_input()
            self.controller.step(*inpt)
        self.controller.update()

        assert self.network._update.call_count == 1
        self.controller._record_update_metrics.assert_called_once_with(critic_loss, actor_loss)

    def test_log(self):
        step = np.random.randint(10) + 1
        self.metrics.get = MagicMock(return_value=step)
        self.metrics.log_metric = MagicMock()

        self.controller.log()

        self.metrics.get.assert_called_once_with('step')
        assert self.metrics.log_metric.call_count == 3
        assert tuple(self.metrics.log_metric.call_args_list[0])[0] == ('reward', step)
        assert tuple(self.metrics.log_metric.call_args_list[1])[0] == ('critic_loss', step)
        assert tuple(self.metrics.log_metric.call_args_list[2])[0] == ('actor_loss', step)

    def test_stop_episode(self):
        self.buffer.add = MagicMock()
        self.metrics.add = MagicMock()
        self.noise.reset = MagicMock()

        inpt = make_input()
        self.controller.stop_episode(inpt[0], inpt[1], inpt[3])

        assert np.all(self.buffer.add.call_args[0][0] == inpt[0])
        assert np.all(self.buffer.add.call_args[0][1] == np.zeros(4))
        assert self.buffer.add.call_args[0][2] == inpt[1]
        assert self.buffer.add.call_args[0][3] == 1.0
        self.metrics.add.assert_called_once_with('reward', inpt[3]['reward'])
        assert self.noise.reset.call_count == 1
