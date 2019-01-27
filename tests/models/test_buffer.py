import numpy as np
import unittest
import pytest

from mvc.models.buffer import Buffer


def make_inputs():
    obs = np.random.random(10)
    action = np.random.random(4)
    reward = np.random.random()
    done = np.random.randint(2)
    return obs, action, reward, done


class BufferTest(unittest.TestCase):
    def test_add(self):
        buffer = Buffer()

        buffer.add(*make_inputs())
        assert buffer.size() == 1

        buffer.add(*make_inputs())
        assert buffer.size() == 2

    def test_capacity(self):
        buffer = Buffer(2)
        buffer.add(*make_inputs())
        buffer.add(*make_inputs())
        buffer.add(*make_inputs())
        assert buffer.size() == 2

    def test_reset(self):
        buffer = Buffer()
        buffer.add(*make_inputs())
        buffer.add(*make_inputs())
        buffer.reset()
        assert buffer.size() == 0

    def test_fetch(self):
        buffer = Buffer()
        obs1, action1, reward1, done1 = make_inputs()
        obs2, action2, reward2, done2 = make_inputs()

        buffer.add(obs1, action1, reward1, done1)
        buffer.add(obs2, action2, reward2, done2)

        batch = buffer.fetch(1)
        assert np.all(batch['obs_t'][0] == obs1)
        assert np.all(batch['obs_tp1'][0] == obs2)
        assert np.all(batch['actions_t'][0] == action1)
        assert batch['rewards_tp1'][0] == reward2
        assert batch['dones_tp1'][0] == done2

    def test_fetch_with_exception(self):
        buffer = Buffer()
        buffer.add(*make_inputs())
        with pytest.raises(AssertionError):
            buffer.fetch(2)
