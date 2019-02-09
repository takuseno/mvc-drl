import numpy as np
import unittest

from mvc.models.buffer import Buffer
from mvc.controllers.sac import SACController
from unittest.mock import MagicMock
from tests.test_utils import DummyNetwork, DummyNoise, DummyMetrics


class SACControllerTest(unittest.TestCase):
    def setUp(self):
        self.network = DummyNetwork()
        self.buffer = Buffer()
        self.metrics = DummyMetrics()
        self.noise = DummyNoise()
        self.controller = SACController(self.network, self.buffer,
                                         self.metrics, self.noise,
                                         num_actions=4, batch_size=32)

    def test_register_metrics(self):
        self.metrics.register = MagicMock()

        self.controller._register_metrics()

        assert tuple(self.metrics.register.call_args_list[0])[0] == ('step', 'single')
        assert tuple(self.metrics.register.call_args_list[1])[0] == ('pi_loss', 'queue')
        assert tuple(self.metrics.register.call_args_list[2])[0] == ('v_loss', 'queue')
        assert tuple(self.metrics.register.call_args_list[3])[0] == ('q1_loss', 'queue')
        assert tuple(self.metrics.register.call_args_list[4])[0] == ('q2_loss', 'queue')
        assert tuple(self.metrics.register.call_args_list[5])[0] == ('reward', 'queue')

    def test_record_update_metrics(self):
        v_loss = np.random.random()
        q_loss = np.random.random(2)
        pi_loss = np.random.random()
        self.metrics.add = MagicMock()

        self.controller._record_update_metrics(v_loss, q_loss, pi_loss)

        assert tuple(self.metrics.add.call_args_list[0])[0] == ('v_loss', v_loss)
        assert tuple(self.metrics.add.call_args_list[1])[0] == ('q1_loss', q_loss[0])
        assert tuple(self.metrics.add.call_args_list[2])[0] == ('q2_loss', q_loss[1])
        assert tuple(self.metrics.add.call_args_list[3])[0] == ('pi_loss', pi_loss)

    def test_log(self):
        step = np.random.randint(10)
        self.metrics.get = MagicMock(return_value=step)
        self.metrics.log_metric = MagicMock()

        self.controller.log()

        self.metrics.get.assert_called_once_with('step')
        assert tuple(self.metrics.log_metric.call_args_list[0])[0] == ('reward', step)
        assert tuple(self.metrics.log_metric.call_args_list[1])[0] == ('v_loss', step)
        assert tuple(self.metrics.log_metric.call_args_list[2])[0] == ('q1_loss', step)
        assert tuple(self.metrics.log_metric.call_args_list[3])[0] == ('q2_loss', step)
        assert tuple(self.metrics.log_metric.call_args_list[4])[0] == ('pi_loss', step)
