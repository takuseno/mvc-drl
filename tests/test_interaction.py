import numpy as np

from unittest import TestCase
from unittest.mock import MagicMock
from mvc.interaction import step, initial_inputs, batch_loop
from mvc.interaction import interact, loop
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

def make_single_inputs():
    obs = np.random.random((16,))
    reward = np.random.random()
    info = {}
    done = np.random.randint(2) == 1.0
    return obs, reward, done, info

class InitialInputsTest(TestCase):
    def test_initial_inputs(self):
        env = DummyEnv()

        reset_obs = np.random.random((4, 84, 84))
        env.reset = MagicMock(return_value=reset_obs)

        obs, reward, done, info = initial_inputs(env)
        assert np.all(obs == reset_obs)
        assert np.all(reward == np.zeros((4,)))
        assert np.all((done == np.zeros(4,)))
        assert info == {}

class StepTest(TestCase):
    def test_step(self):
        env = DummyEnv()
        view = DummyView()

        action = np.random.random((4, 10))
        inputs = make_inputs()
        outputs = make_inputs()
        view.step = MagicMock(return_value=action)
        env.step = MagicMock(return_value=outputs)

        obs, reward, done, info = step(env, view, *inputs)
        assert np.all(outputs[0] == obs)
        assert np.all(outputs[1] == reward)
        assert np.all(outputs[2] == done)
        assert outputs[3] == info

        assert np.all(view.step.call_args[0][0] == inputs[0])
        assert np.all(view.step.call_args[0][1] == inputs[1])
        assert np.all(view.step.call_args[0][2] == inputs[2])
        assert view.step.call_args[0][3] == inputs[3]

        assert np.all(env.step.call_args[0][0] == action)

class BatchLoopTest(TestCase):
    def test_batch_loop(self):
        env = DummyEnv()
        view = DummyView()

        obs = make_inputs()[0]
        env.reset = MagicMock(return_value=obs)
        outputs = make_inputs()
        env.step = MagicMock(return_value=outputs)

        view.step = MagicMock(return_value=np.random.random())
        view.is_finished = MagicMock(side_effect=lambda: env.step.call_count == 5)
        view.should_eval = MagicMock(return_value=False)

        hook = MagicMock()

        batch_loop(env, view, hook)

        assert env.step.call_count == 5
        assert view.is_finished.call_count == 5
        assert np.all(view.step.call_args[0][0] == outputs[0])
        assert np.all(view.step.call_args[0][1] == outputs[1])
        assert np.all(view.step.call_args[0][2] == outputs[2])
        assert view.step.call_args[0][3] == outputs[3]
        assert hook.call_count == 5


class LoopTest(TestCase):
    def test_loop(self):
        env = DummyEnv()
        view = DummyView()

        obs = make_single_inputs()[0]
        env.reset = MagicMock(return_value=obs)
        outputs = make_single_inputs()
        outputs = outputs[:2] + (False,) + (outputs[-1],)
        env.step = MagicMock(return_value=outputs)

        view.step = MagicMock(return_value=np.random.random())
        view.is_finished = MagicMock(side_effect=lambda: env.step.call_count == 5)
        view.should_eval = MagicMock(return_value=False)

        hook = MagicMock()

        loop(env, view, hook)

        assert env.step.call_count == 5
        assert view.is_finished.call_count == 5
        assert np.all(view.step.call_args[0][0] == outputs[0])
        assert np.all(view.step.call_args[0][1] == outputs[1])
        assert np.all(view.step.call_args[0][2] == outputs[2])
        assert view.step.call_args[0][3] == outputs[3]
        assert hook.call_count == 5

    def test_loop_with_stop_episode(self):
        env = DummyEnv()
        view = DummyView()

        obs = make_single_inputs()[0]
        env.reset = MagicMock(return_value=obs)
        outputs = make_single_inputs()
        outputs = outputs[:2] + (True,) + (outputs[-1],)
        env.step = MagicMock(return_value=outputs)

        view.step = MagicMock(return_value=np.random.random())
        view.is_finished = MagicMock(side_effect=lambda: env.step.call_count == 5)
        view.should_eval = MagicMock(return_value=False)
        view.stop_episode = MagicMock()

        hook = MagicMock()

        loop(env, view, hook)

        assert view.stop_episode.call_count == 4


class InteractionTest(TestCase):
    def test_loop_with_eval(self):
        env = DummyEnv()
        view = DummyView()
        eval_env = DummyEnv()
        eval_view = DummyView()

        obs = make_single_inputs()[0]
        eval_env.reset = env.reset = MagicMock(return_value=obs)
        outputs = make_single_inputs()
        eval_env.step = env.step = MagicMock(return_value=outputs)
        view.is_finished = MagicMock(side_effect=lambda: env.step.call_count == 5)
        view.should_eval = MagicMock(side_effect=lambda: env.step.call_count == 2)
        eval_view.is_finished = MagicMock(side_effect=lambda: eval_env.step.call_count == 5)
        eval_view.should_eval = MagicMock(side_effect=lambda: eval_env.step.call_count == 2)

        interact(env, view, eval_env, eval_view, batch=False)

        assert env.step.call_count == 5
        assert eval_env.step.call_count == 5

    def test_batch_loop_with_eval(self):
        env = DummyEnv()
        view = DummyView()
        eval_env = DummyEnv()
        eval_view = DummyView()

        obs = make_inputs()[0]
        eval_env.reset = env.reset = MagicMock(return_value=obs)
        outputs = make_inputs()
        eval_env.step = env.step = MagicMock(return_value=outputs)
        view.is_finished = MagicMock(side_effect=lambda: env.step.call_count == 5)
        view.should_eval = MagicMock(side_effect=lambda: env.step.call_count == 2)
        eval_view.is_finished = MagicMock(side_effect=lambda: eval_env.step.call_count == 5)
        eval_view.should_eval = MagicMock(side_effect=lambda: eval_env.step.call_count == 2)

        interact(env, view, eval_env, eval_view, batch=True)

        assert env.step.call_count == 5
        assert eval_env.step.call_count == 5
