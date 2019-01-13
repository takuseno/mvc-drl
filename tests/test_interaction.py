import numpy as np

from unittest import TestCase
from unittest.mock import MagicMock
from mvc.interaction import BatchInteraction
from mvc.view import View


class DummyEnv:
    def step(self, obs, reward, done, info):
        pass

    def reset(self):
        pass

class DummyView(View):
    def __init__(self):
        pass

    def step(self, obs, reward, done, info):
        pass

    def is_finished(self):
        pass

    def stop_episode(self, obs, reward, info):
        pass

def make_inputs():
    obs = np.random.random((4, 84, 84))
    reward = np.random.random((4,))
    info = [{} for _ in range(4)]
    done = np.array([np.random.randint(2) for _ in range(4)])
    return obs, reward, done, info

class BatchInteractionTest(TestCase):
    def test_step(self):
        env = DummyEnv()
        view = DummyView()
        interaction = BatchInteraction(env, view)

        action = np.random.random((4, 10))
        inputs = make_inputs()
        outputs = make_inputs()
        view.step = MagicMock(return_value=action)
        env.step = MagicMock(return_value=outputs)

        obs, reward, done, info = interaction.step(*inputs)
        assert np.all(outputs[0] == obs)
        assert np.all(outputs[1] == reward)
        assert np.all(outputs[2] == done)
        assert outputs[3] == info

        assert np.all(view.step.call_args[0][0] == inputs[0])
        assert np.all(view.step.call_args[0][1] == inputs[1])
        assert np.all(view.step.call_args[0][2] == inputs[2])
        assert view.step.call_args[0][3] == inputs[3]

        assert np.all(env.step.call_args[0][0] == action)

    def test_loop(self):
        env = DummyEnv()
        view = DummyView()
        interaction = BatchInteraction(env, view)

        obs = make_inputs()[0]
        env.reset = MagicMock(return_value=obs)
        outputs = make_inputs()
        interaction.step = MagicMock(return_value=outputs)
        view.is_finished = MagicMock(side_effect=lambda: interaction.step.call_count == 5)

        interaction.loop()

        assert interaction.step.call_count == 5
        assert view.is_finished.call_count == 5
        assert np.all(interaction.step.call_args[0][0] == outputs[0])
        assert np.all(interaction.step.call_args[0][1] == outputs[1])
        assert np.all(interaction.step.call_args[0][2] == outputs[2])
        assert interaction.step.call_args[0][3] == outputs[3]
