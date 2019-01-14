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

    def should_eval(self):
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

        obs, reward, done, info = interaction.step(env, view, *inputs)
        assert np.all(outputs[0] == obs)
        assert np.all(outputs[1] == reward)
        assert np.all(outputs[2] == done)
        assert outputs[3] == info

        assert np.all(view.step.call_args[0][0] == inputs[0])
        assert np.all(view.step.call_args[0][1] == inputs[1])
        assert np.all(view.step.call_args[0][2] == inputs[2])
        assert view.step.call_args[0][3] == inputs[3]

        assert np.all(env.step.call_args[0][0] == action)

    def test_initial_inputs(self):
        env = DummyEnv()
        view = DummyView()
        interaction = BatchInteraction(env, view)

        reset_obs = np.random.random((4, 84, 84))
        env.reset = MagicMock(return_value=reset_obs)

        obs, reward, done, info = interaction.initial_inputs(env)
        assert np.all(obs == reset_obs)
        assert np.all(reward == np.zeros((4,)))
        assert np.all((done == np.zeros(4,)))
        assert info == {}

    def test_loop(self):
        env = DummyEnv()
        view = DummyView()
        interaction = BatchInteraction(env, view)

        obs = make_inputs()[0]
        env.reset = MagicMock(return_value=obs)
        outputs = make_inputs()
        interaction.step = MagicMock(return_value=outputs)
        interaction.eval_loop = MagicMock()
        view.is_finished = MagicMock(side_effect=lambda: interaction.step.call_count == 5)
        view.should_eval = MagicMock(return_value=False)

        interaction.loop()

        assert interaction.step.call_count == 5
        assert view.is_finished.call_count == 5
        assert np.all(interaction.step.call_args[0][2] == outputs[0])
        assert np.all(interaction.step.call_args[0][3] == outputs[1])
        assert np.all(interaction.step.call_args[0][4] == outputs[2])
        assert interaction.step.call_args[0][5] == outputs[3]
        interaction.eval_loop.assert_not_called()

    def test_loop_with_eval(self):
        env = DummyEnv()
        view = DummyView()
        eval_env = DummyEnv()
        eval_view = DummyView()
        interaction = BatchInteraction(env, view, eval_env, eval_view)

        obs = make_inputs()[0]
        env.reset = MagicMock(return_value=obs)
        outputs = make_inputs()
        interaction.step = MagicMock(return_value=outputs)
        interaction.eval_loop = MagicMock()
        view.is_finished = MagicMock(side_effect=lambda: interaction.step.call_count == 5)
        view.should_eval = MagicMock(side_effect=lambda: interaction.step.call_count == 2)

        interaction.loop()

        print(interaction.eval_loop.call_count)
        interaction.eval_loop.assert_called_once()

    def test_eval_loop(self):
        env = DummyEnv()
        view = DummyView()
        eval_env = DummyEnv()
        eval_view = DummyView()
        interaction = BatchInteraction(env, view, eval_env, eval_view)

        obs = make_inputs()[0]
        eval_env.reset = MagicMock(return_value=obs)
        outputs = make_inputs()
        interaction.step = MagicMock(return_value=outputs)
        eval_view.is_finished = MagicMock(side_effect=lambda: interaction.step.call_count == 5)

        interaction.eval_loop()

        assert interaction.step.call_count == 5
        assert eval_view.is_finished.call_count == 5
        assert np.all(interaction.step.call_args[0][2] == outputs[0])
        assert np.all(interaction.step.call_args[0][3] == outputs[1])
        assert np.all(interaction.step.call_args[0][4] == outputs[2])
        assert interaction.step.call_args[0][5] == outputs[3]
