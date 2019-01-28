import numpy as np
import pytest
import unittest

from mvc.preprocess import compute_returns, compute_gae
from mvc.models.rollout import Rollout


def make_inputs():
    return {
        'obs_t': np.random.random((4, 84, 84)),
        'action_t': np.random.random((4, 4)),
        'reward_t': np.random.random((4,)),
        'value_t': np.random.random((4,)),
        'log_prob_t': np.random.random((4,)),
        'terminal_t': np.random.random((4,))
    }

def insert_inputs_to_rollout(inputs, rollout):
    rollout.add(inputs['obs_t'], inputs['action_t'], inputs['reward_t'],
                inputs['value_t'], inputs['log_prob_t'], inputs['terminal_t'])

def assert_inputs_with_rollout(inputs, rollout, index):
    assert np.all(inputs['obs_t'] == rollout.obs_t[index])
    assert np.all(inputs['action_t'] == rollout.actions_t[index])
    assert np.all(inputs['reward_t'] == rollout.rewards_t[index])
    assert np.all(inputs['value_t'] == rollout.values_t[index])
    assert np.all(inputs['log_prob_t'] == rollout.log_probs_t[index])
    assert np.all(inputs['terminal_t'] == rollout.terminals_t[index])

class RolloutTest(unittest.TestCase):
    def test_add_success(self):
        rollout = Rollout()
        inputs1 = make_inputs()
        insert_inputs_to_rollout(inputs1, rollout)
        assert_inputs_with_rollout(inputs1, rollout, 0)

        inputs2 = make_inputs()
        insert_inputs_to_rollout(inputs2, rollout)
        assert_inputs_with_rollout(inputs1, rollout, 0)
        assert_inputs_with_rollout(inputs2, rollout, 1)

    def test_add_with_dimension_mismatch(self):
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['action_t'] = np.random.random((5, 4))
            insert_inputs_to_rollout(inputs, rollout)
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['reward_t'] = np.random.random((5,))
            insert_inputs_to_rollout(inputs, rollout)
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['value_t'] = np.random.random((5,))
            insert_inputs_to_rollout(inputs, rollout)
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['log_prob_t'] = np.random.random((5,))
            insert_inputs_to_rollout(inputs, rollout)
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['terminal_t'] = np.random.random((5,))
            insert_inputs_to_rollout(inputs, rollout)

    def test_add_with_shape_error(self):
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['reward_t'] = np.random.random((4, 5))
            insert_inputs_to_rollout(inputs, rollout)
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['value_t'] = np.random.random((4, 5))
            insert_inputs_to_rollout(inputs, rollout)
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['log_prob_t'] = np.random.random((4, 5))
            insert_inputs_to_rollout(inputs, rollout)
        with pytest.raises(AssertionError):
            rollout = Rollout()
            inputs = make_inputs()
            inputs['terminal_t'] = np.random.random((4, 5))
            insert_inputs_to_rollout(inputs, rollout)

    def test_fetch(self):
        gamma = np.random.random()
        lam = np.random.random()
        rollout = Rollout()
        inputs1 = make_inputs()
        insert_inputs_to_rollout(inputs1, rollout)

        with pytest.raises(AssertionError):
            rollout.fetch(gamma, lam)

        inputs2 = make_inputs()
        insert_inputs_to_rollout(inputs2, rollout)
        trajectory = rollout.fetch(gamma, lam)

        assert np.all(inputs1['obs_t'] == trajectory['obs_t'][0])
        assert np.all(inputs1['action_t'] == trajectory['actions_t'][0])
        assert np.all(inputs1['log_prob_t'] == trajectory['log_probs_t'][0])
        assert trajectory['returns_t'].shape == (1, 4)
        assert trajectory['advantages_t'].shape == (1, 4)
    
    def test_size(self):
        rollout = Rollout()
        self.assertEqual(rollout.size(), 0)
        inputs = make_inputs()
        insert_inputs_to_rollout(inputs, rollout)
        self.assertEqual(rollout.size(), 1)
        insert_inputs_to_rollout(inputs, rollout)
        self.assertEqual(rollout.size(), 2)
