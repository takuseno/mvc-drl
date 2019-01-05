import unittest

from unittest.mock import MagicMock, Mock
from mvc.view import TrainView, EvalView


class DummyController:
    def step(self, obs, reward, done):
        pass

    def stop_episode(self, obs, reward):
        pass

    def should_update(self):
        pass

    def update(self):
        pass


class TrainViewTest(unittest.TestCase):
    def test_step_without_update(self):
        controller = DummyController()
        view = TrainView(controller)
        controller.should_update = MagicMock(return_value=False)
        controller.update = MagicMock(side_effect=Exception)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done')

    def test_step_with_update(self):
        controller = DummyController()
        view = TrainView(controller)
        controller.should_update = MagicMock(return_value=True)
        controller.update = MagicMock(unsafe=True)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done')
        controller.update.assert_called()

    def test_stop_episode(self):
        controller = DummyController()
        view = TrainView(controller)
        controller.stop_episode = MagicMock()

        self.assertEqual(view.stop_episode('obs', 'reward'), None)
        controller.stop_episode.assert_called_once_with('obs', 'reward')

class EvalViewTest(unittest.TestCase):
    def test_step_without_should_update_false(self):
        controller = DummyController()
        view = EvalView(controller)
        controller.should_update = MagicMock(return_value=False)
        controller.update = MagicMock(side_effect=Exception)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done')

    def test_step_with_should_update_true(self):
        controller = DummyController()
        view = EvalView(controller)
        controller.should_update = MagicMock(return_value=True)
        controller.update = MagicMock(side_effect=Exception)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done')

    def test_stop_episode(self):
        controller = DummyController()
        view = EvalView(controller)
        controller.stop_episode = MagicMock()

        self.assertEqual(view.stop_episode('obs', 'reward'), None)
        controller.stop_episode.assert_called_once_with('obs', 'reward')
