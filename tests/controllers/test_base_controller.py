import numpy as np
import pytest

from unittest import TestCase
from unittest.mock import MagicMock
from mvc.controllers.base_controller import BaseController
from mvc.models.metrics import Metrics


class BaseControllerTest(TestCase):
    def test_step(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        with pytest.raises(NotImplementedError):
            controller.step('obs', 'reward', 'done', 'info')

    def test_should_update(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        with pytest.raises(NotImplementedError):
            controller.should_update()

    def test_update(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        with pytest.raises(NotImplementedError):
            controller.update()

    def test_stop_episode(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        with pytest.raises(NotImplementedError):
            controller.stop_episode('obs', 'reward', 'info')

    def test_should_log(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        metrics.get = MagicMock(return_value=np.random.randint(100))
        assert not controller.should_log()
        metrics.get.assert_called_once_with('step')

        metrics.get = MagicMock(return_value=np.random.randint(100) * 120)
        assert controller.should_log()
        metrics.get.assert_called_once_with('step')

    def test_log(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        with pytest.raises(NotImplementedError):
            controller.log()

    def test_should_save(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        metrics.get = MagicMock(return_value=np.random.randint(100))
        assert not controller.should_save()
        metrics.get.assert_called_once_with('step')

        metrics.get = MagicMock(return_value=np.random.randint(100) * 130)
        assert controller.should_save()
        metrics.get.assert_called_once_with('step')

    def test_save(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        step = np.random.randint(100)
        metrics.get = MagicMock(return_value=step)
        metrics.save_model = MagicMock()
        controller.save()
        metrics.get.assert_called_once_with('step')
        metrics.save_model.assert_called_once_with(step)

    def test_if_finished(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        metrics.get = MagicMock(return_value=np.random.randint(100))
        assert not controller.is_finished()
        metrics.get.assert_called_once_with('step')

        metrics.get = MagicMock(return_value=np.random.randint(100) * 110)
        assert controller.is_finished()
        metrics.get.assert_called_once_with('step')

    def test_should_eval(self):
        metrics = Metrics('test')
        controller = BaseController(metrics, 110, 120, 130, 140)

        metrics.get = MagicMock(return_value=np.random.randint(100))
        assert not controller.should_eval()
        metrics.get.assert_called_once_with('step')

        metrics.get = MagicMock(return_value=np.random.randint(100) * 140)
        assert controller.should_eval()
        metrics.get.assert_called_once_with('step')
