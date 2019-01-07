import unittest

from mvc.actions import Action


class ActionTest(unittest.TestCase):
    def test_properties(self):
        action = Action('action', 'log_prob', 'value')
        self.assertEqual(action.action, 'action')
        self.assertEqual(action.value, 'value')
        self.assertEqual(action.log_prob, 'log_prob')
