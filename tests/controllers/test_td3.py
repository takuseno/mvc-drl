import numpy as np
import unittest

from mvc.models.buffer import Buffer
from mvc.controllers.td3 import TD3Controller
from unittest.mock import MagicMock
from tests.test_utils import DummyNetwork, DummyNoise, DummyMetrics


class TD3ControllerTest(unittest.TestCase):
    def setUp(self):
        self.network = DummyNetwork()
        self.buffer = Buffer()
        self.metrics = DummyMetrics()
        self.noise = DummyNoise()
        self.controller = TD3Controller(self.network, self.buffer,
                                        self.metrics, self.noise,
                                        num_actions=4, batch_size=32)

    def test_record_update_metrics(self):
        critic_loss = np.random.random()
        actor_loss = np.random.random()
        self.metrics.add = MagicMock()

        self.controller._record_update_metrics(critic_loss, actor_loss)

        assert tuple(self.metrics.add.call_args_list[0])[0] == ('critic_loss', critic_loss)
        assert tuple(self.metrics.add.call_args_list[1])[0] == ('actor_loss', actor_loss)

    def test_record_update_metrics_with_none_actor_loss(self):
        critic_loss = np.random.random()
        actor_loss = None
        self.metrics.add = MagicMock()

        self.controller._record_update_metrics(critic_loss, actor_loss)

        self.metrics.add.assert_called_once_with('critic_loss', critic_loss)

    def test_update(self):
        critic_loss = np.random.random()
        actor_loss = np.random.random()
        self.network._update = MagicMock(return_value=(critic_loss, actor_loss))
        self.network._update_arguments = MagicMock(return_value=['obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1', 'update_actor'])
        self.controller._record_update_metrics = MagicMock()
        self.controller.should_update = MagicMock(return_value=True)
        self.buffer.fetch = MagicMock(return_value={
            'obs_t': np.random.random((32, 100)),
            'actions_t': np.random.random((32, 4)),
            'rewards_tp1': np.random.random((32,)),
            'obs_tp1': np.random.random((32, 100)),
            'dones_tp1': np.random.random((32,))
        })

        self.metrics.get = MagicMock(return_value=2)
        self.controller.update()

        assert self.network._update.call_count == 1
        self.controller._record_update_metrics.assert_called_once_with(critic_loss, actor_loss)
        print(tuple(self.network._update.call_args_list[0]))
        assert tuple(self.network._update.call_args_list[0])[1]['update_actor'] == True

        self.metrics.get = MagicMock(return_value=1)
        self.controller.update()
        assert tuple(self.network._update.call_args_list[1])[1]['update_actor'] == False
