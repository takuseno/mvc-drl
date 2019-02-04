import unittest

from unittest.mock import MagicMock, Mock
from mvc.view import View


class DummyController:
    def step(self, obs, reward, done):
        pass

    def stop_episode(self, obs, reward):
        pass

    def should_update(self):
        pass

    def update(self):
        pass

    def should_log(self):
        pass

    def log(self):
        pass

    def should_save(self):
        pass

    def save(self):
        pass

class ViewTest(unittest.TestCase):
    def test_step_without_update(self):
        controller = DummyController()
        view = View(controller)
        controller.should_update = MagicMock(return_value=False)
        controller.update = MagicMock(side_effect=Exception)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done', 'info'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done', 'info')

    def test_step_with_update(self):
        controller = DummyController()
        view = View(controller)
        controller.should_update = MagicMock(return_value=True)
        controller.update = MagicMock(unsafe=True)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done', 'info'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done', 'info')
        controller.update.assert_called()

    def test_step_without_log(self):
        controller = DummyController()
        view = View(controller)
        controller.should_log = MagicMock(return_value=False)
        controller.log = MagicMock(side_effect=Exception)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done', 'info'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done', 'info')
        controller.log.assert_not_called()

    def test_step_with_log(self):
        controller = DummyController()
        view = View(controller)
        controller.should_log = MagicMock(return_value=True)
        controller.log = MagicMock(unsafe=True)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done', 'info'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done', 'info')
        assert controller.log.call_count == 1

    def test_step_without_save(self):
        controller = DummyController()
        view = View(controller)
        controller.should_save = MagicMock(return_value=False)
        controller.save = MagicMock(unsafe=True)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done', 'info'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done', 'info')
        controller.save.assert_not_called()

    def test_step_without_save(self):
        controller = DummyController()
        view = View(controller)
        controller.should_save = MagicMock(return_value=True)
        controller.save = MagicMock(unsafe=True)
        controller.step = MagicMock(return_value='action')

        self.assertEqual(view.step('obs', 'reward', 'done', 'info'), 'action')
        controller.step.assert_called_once_with('obs', 'reward', 'done', 'info')
        controller.save.assert_called_once_with()

    def test_stop_episode(self):
        controller = DummyController()
        view = View(controller)
        controller.stop_episode = MagicMock()

        self.assertEqual(view.stop_episode('obs', 'reward', 'info'), None)
        controller.stop_episode.assert_called_once_with('obs', 'reward', 'info')

    def test_is_finished(self):
        controller = DummyController()
        view = View(controller)
        controller.save = MagicMock()

        controller.is_finished = MagicMock(return_value=False)
        assert not view.is_finished()
        controller.save.assert_not_called()

        controller.is_finished = MagicMock(return_value=True)
        assert view.is_finished()
        controller.save.assert_called_once_with()

    def test_should_eval(self):
        controller = DummyController()
        view = View(controller)

        controller.should_eval = MagicMock(return_value=False)
        assert not view.should_eval()

        controller.should_eval = MagicMock(return_value=True)
        assert view.should_eval()
