import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import pytest

from mvc.models.networks.ppo import PPONetwork
from mvc.models.networks.ppo import build_value_loss
from mvc.models.networks.ppo import build_policy_loss
from mvc.models.networks.ppo import build_entropy_loss

from tests.test_utils import make_tf_inpt, make_fcs, to_tf
from tests.test_utils import assert_hidden_variable_shape, assert_variable_mismatch


class BuildValueLossTest(tf.test.TestCase):
    def test_success_with_unclipped(self):
        epsilon = np.random.random()
        nd_returns = np.random.random((4, 1))
        nd_old_values = np.random.random((4, 1))
        nd_values = nd_old_values + epsilon * 0.9
        values = to_tf(nd_values)
        returns = to_tf(nd_returns)
        old_values = to_tf(nd_old_values)

        loss = build_value_loss(values, returns, old_values, epsilon, 0.5)
        assert len(loss.shape) == 0

        with self.test_session() as sess:
            answer = 0.5 * np.mean((nd_returns - nd_values) ** 2)
            assert np.allclose(sess.run(loss), answer)

    def test_success_with_positive_clipped(self):
        epsilon = np.random.random()
        nd_old_values = np.random.random((4, 1))
        nd_values = nd_old_values + epsilon * 1.1
        nd_returns = np.random.random((4, 1)) + nd_values
        values = to_tf(nd_values)
        returns = to_tf(nd_returns)
        old_values = to_tf(nd_old_values)

        loss = build_value_loss(values, returns, old_values, epsilon, 0.5)

        with self.test_session() as sess:
            answer = 0.5 * np.mean((nd_returns - (nd_old_values + epsilon)) ** 2)
            assert np.allclose(sess.run(loss), answer)

    def test_success_with_negative_clipped(self):
        epsilon = np.random.random()
        nd_old_values = np.random.random((4, 1))
        nd_values = nd_old_values - epsilon * 1.1
        nd_returns = -np.random.random((4, 1))
        values = to_tf(nd_values)
        returns = to_tf(nd_returns)
        old_values = to_tf(nd_old_values)

        loss = build_value_loss(values, returns, old_values, epsilon, 0.5)

        with self.test_session() as sess:
            answer = 0.5 * np.mean((nd_returns - (nd_old_values - epsilon)) ** 2)
            assert np.allclose(sess.run(loss), answer)


class BuildEntropyLossTest(tf.test.TestCase):
    def test_sccess(self):
        loc = np.random.random((4, 2))
        scale = np.random.random((4, 2))
        dist = tf.distributions.Normal(loc=loc, scale=scale)
        entropy = dist.entropy()
        loss = build_entropy_loss(dist, 0.01)

        with self.test_session() as sess:
            answer = -np.mean(0.01 * sess.run(entropy))
            assert np.allclose(sess.run(loss), answer)


class BuildPolicyLossTest(tf.test.TestCase):
    def setUp(self):
        self.num_actions = np.random.randint(5) + 1

    def test_success_with_positive_not_clipped(self):
        # new/old < 0.5
        nd_log_probs = np.log(np.random.random((4, 1)) * 0.2)
        nd_old_log_probs = np.log(np.random.random((4, 1)) * 0.5 + 0.5)
        nd_advantages = np.random.random((4, 1))

        log_probs = to_tf(nd_log_probs)
        old_log_probs = to_tf(nd_old_log_probs)
        advantages = to_tf(nd_advantages)

        loss = build_policy_loss(log_probs, old_log_probs, advantages, 0.2)

        ratio = np.exp(nd_log_probs - nd_old_log_probs)
        answer = -np.mean(ratio * nd_advantages)

        with self.test_session() as sess:
            assert np.allclose(sess.run(loss), answer)

    def test_success_with_positive_clipped(self):
        # new/old > 2.0
        nd_log_probs = np.log(np.random.random((4, 1)) * 0.5 + 0.5)
        nd_old_log_probs = np.log(np.random.random((4, 1)) * 0.2)
        nd_advantages = np.random.random((4, 1))

        log_probs = to_tf(nd_log_probs)
        old_log_probs = to_tf(nd_old_log_probs)
        advantages = to_tf(nd_advantages)

        loss = build_policy_loss(log_probs, old_log_probs, advantages, 0.2)

        ratio = np.exp(nd_log_probs - nd_old_log_probs)
        ratio[ratio > 1.2] = 1.2
        answer = -np.mean(ratio * nd_advantages)

        with self.test_session() as sess:
            assert np.allclose(sess.run(loss), answer)

    def test_success_with_positive_clipped(self):
        # new/old > 2.0
        nd_log_probs = np.log(np.random.random((4, 1)) * 0.5 + 0.5)
        nd_old_log_probs = np.log(np.random.random((4, 1)) * 0.2)
        nd_advantages = -np.random.random((4, 1))

        log_probs = to_tf(nd_log_probs)
        old_log_probs = to_tf(nd_old_log_probs)
        advantages = to_tf(nd_advantages)

        loss = build_policy_loss(log_probs, old_log_probs, advantages, 0.2)

        ratio = np.exp(nd_log_probs - nd_old_log_probs)
        answer = -np.mean(ratio * nd_advantages)

        with self.test_session() as sess:
            assert np.allclose(sess.run(loss), answer)

    def test_success_with_negative_clipped(self):
        # new/old < 0.5
        nd_log_probs = np.log(np.random.random((4, 1)) * 0.2)
        nd_old_log_probs = np.log(np.random.random((4, 1)) * 0.5 + 0.5)
        nd_advantages = -np.random.random((4, 1))

        log_probs = to_tf(nd_log_probs)
        old_log_probs = to_tf(nd_old_log_probs)
        advantages = to_tf(nd_advantages)

        loss = build_policy_loss(log_probs, old_log_probs, advantages, 0.2)

        ratio = np.exp(nd_log_probs - nd_old_log_probs)
        ratio[ratio < 0.8] = 0.8
        answer = -np.mean(ratio * nd_advantages)

        with self.test_session() as sess:
            assert np.allclose(sess.run(loss), answer)


class PPONetworkTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.state_shape = [np.random.randint(4) + 1]
        self.num_envs = np.random.randint(4) + 1
        self.num_actions = np.random.randint(4) + 1
        self.batch_size = np.random.randint(20) + 1
        self.epsilon = np.random.random()
        self.lr = np.random.random()
        self.grad_clip = np.random.random()
        self.value_factor = np.random.random()
        self.entropy_factor = np.random.random()
        self.network = PPONetwork([64, 64], self.state_shape,
                                  self.num_envs, self.num_actions,
                                  self.batch_size, self.epsilon, self.lr,
                                  self.grad_clip, self.value_factor,
                                  self.entropy_factor)

    def test_build(self):
        assert int(self.network.action.shape[0]) == self.num_envs
        assert int(self.network.action.shape[1]) == self.num_actions

        assert len(self.network.log_policy.shape) == 1
        assert int(self.network.log_policy.shape[0]) == self.num_envs

        assert len(self.network.value.shape) == 1
        assert int(self.network.value.shape[0]) == self.num_envs

        assert len(self.network.loss.shape) == 0

    def test_infer_arguments(self):
        args = self.network._infer_arguments()
        keys = ['obs_t']
        for key in args:
            assert key in keys

    def test_update_arguments(self):
        args = self.network._update_arguments()
        keys = ['obs_t', 'actions_t', 'log_probs_t', 'returns_t', 'advantages_t', 'values_t']
        for key in args:
            assert key in keys

    # note: to test infer and update, use 'infer' and 'update' instead of
    # '_infer' and '_update'
    def test_infer(self):
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            obs = np.random.random([self.num_envs] + self.state_shape)
            output = self.network.infer(obs_t=obs)

        assert output.action.shape == (self.num_envs, self.num_actions)
        assert output.log_prob.shape == (self.num_envs,)
        assert output.value.shape == (self.num_envs,)

    def test_update(self):
        obs = np.random.random([self.batch_size] + self.state_shape)
        actions = np.random.random((self.batch_size, self.num_actions))
        returns = np.random.random((self.batch_size,))
        returns = np.random.random((self.batch_size,))
        advantages = np.random.random((self.batch_size,))
        old_log_probs = np.random.random((self.batch_size))
        old_values = np.random.random((self.batch_size))
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ppo')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            before = sess.run(variables)

            loss = self.network.update(obs_t=obs, actions_t=actions,
                                       returns_t=returns, advantages_t=advantages,
                                       log_probs_t=old_log_probs,
                                       values_t=old_values)

            after = sess.run(variables)

        for var1, var2 in zip(before, after):
            assert not np.allclose(var1, var2)
