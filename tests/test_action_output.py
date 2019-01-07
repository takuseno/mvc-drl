import unittest

from mvc.action_output import ActionOutput


class ActionOutputTest(unittest.TestCase):
    def test_properties(self):
        output = ActionOutput('action', 'log_prob', 'value')
        self.assertEqual(output.action, 'action')
        self.assertEqual(output.value, 'value')
        self.assertEqual(output.log_prob, 'log_prob')
