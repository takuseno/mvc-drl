import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import numpy as np
import pytest

from mvc.models.networks.ppo import PPONetwork
from mvc.models.networks.ppo import build_value_loss
from mvc.models.networks.ppo import build_policy_loss
from mvc.models.networks.ppo import build_entropy_loss
from mvc.models.networks.ppo import ppo_function

from tests.test_utils import make_tf_inpt, make_fcs
from tests.test_utils import assert_hidden_variable_shape, assert_variable_mismatch


def function(num_actions):
    def func(obs):
        with tf.variable_scope('test'):
            out = tf.layers.dense(obs, 64)
            loc = tf.layers.dense(out, num_actions)
            scale = tf.layers.dense(out, num_actions)
            dist = tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

            out = tf.layers.dense(obs, 64)
            value = tf.layers.dense(out, 1)
        return dist, value
    return func


class PPOFunctionTest(tf.test.TestCase):
    def test_ppo_function(self):
        inpt = make_tf_inpt()
        fcs = make_fcs()
        num_actions = np.random.randint(10) + 1

        func = ppo_function(fcs, num_actions, 'scope')

        policy, value = func(inpt)

        # to check connection
        optimizer = tf.train.AdamOptimizer(1e-4)
        optimize_expr = optimizer.minimize(tf.reduce_mean(policy.sample(1)) + tf.reduce_mean(value))

        assert int(policy.sample(1)[0].shape[0]) == int(inpt.shape[0])
        assert int(policy.sample(1)[0].shape[1]) == num_actions
        assert int(value.shape[0]) == int(inpt.shape[0])
        assert int(value.shape[1]) == 1

        policy_hiddens = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'scope/policy/hiddens')
        assert_hidden_variable_shape(policy_hiddens, inpt, fcs)

        value_hiddens = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'scope/value/hiddens')
        assert_hidden_variable_shape(value_hiddens, inpt, fcs)

        variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'scope')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            before = sess.run(variable)
            before_policy_hiddens = sess.run(policy_hiddens)
            before_value_hiddens = sess.run(value_hiddens)

            sess.run(optimize_expr)

            after = sess.run(variable)
            assert_variable_mismatch(before, after)


class BuildValueLossTest(tf.test.TestCase):
    def test_success_with_unclipped(self):
        epsilon = np.random.random()
        nd_returns = np.random.random((4, 1))
        nd_old_values = np.random.random((4, 1))
        nd_values = nd_old_values + epsilon * 0.9
        values = tf.constant(nd_values)
        returns = tf.constant(nd_returns)
        old_values = tf.constant(nd_old_values)

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
        values = tf.constant(nd_values)
        returns = tf.constant(nd_returns)
        old_values = tf.constant(nd_old_values)

        loss = build_value_loss(values, returns, old_values, epsilon, 0.5)

        with self.test_session() as sess:
            answer = 0.5 * np.mean((nd_returns - (nd_old_values + epsilon)) ** 2)
            assert np.allclose(sess.run(loss), answer)

    def test_success_with_negative_clipped(self):
        epsilon = np.random.random()
        nd_old_values = np.random.random((4, 1))
        nd_values = nd_old_values - epsilon * 1.1
        nd_returns = -np.random.random((4, 1))
        values = tf.constant(nd_values)
        returns = tf.constant(nd_returns)
        old_values = tf.constant(nd_old_values)

        loss = build_value_loss(values, returns, old_values, epsilon, 0.5)

        with self.test_session() as sess:
            answer = 0.5 * np.mean((nd_returns - (nd_old_values - epsilon)) ** 2)
            assert np.allclose(sess.run(loss), answer)

    def test_with_invalid_shape(self):
        values = tf.constant(np.random.random((4)))
        old_values = tf.constant(np.random.random((4, 1)))
        returns = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_value_loss(values, returns, old_values, 0.5, 0.2)

        values = tf.constant(np.random.random((4, 1)))
        old_values = tf.constant(np.random.random((4, 1)))
        returns = tf.constant(np.random.random((4)))
        with pytest.raises(AssertionError):
            build_value_loss(values, returns, old_values, 0.5, 0.2)

        values = tf.constant(np.random.random((4, 1)))
        old_values = tf.constant(np.random.random((4)))
        returns = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_value_loss(values, returns, old_values, 0.5, 0.2)

        values = tf.constant(np.random.random((4, 2)))
        old_values = tf.constant(np.random.random((4, 1)))
        returns = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_value_loss(values, returns, old_values, 0.5, 0.2)

        values = tf.constant(np.random.random((4, 1)))
        old_values = tf.constant(np.random.random((4, 1)))
        returns = tf.constant(np.random.random((4, 2)))
        with pytest.raises(AssertionError):
            build_value_loss(values, returns, old_values, 0.5, 0.2)

        values = tf.constant(np.random.random((4, 1)))
        old_values = tf.constant(np.random.random((4, 2)))
        returns = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_value_loss(values, returns, old_values, 0.5, 0.2)

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

        log_probs = tf.constant(nd_log_probs)
        old_log_probs = tf.constant(nd_old_log_probs)
        advantages = tf.constant(nd_advantages)

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

        log_probs = tf.constant(nd_log_probs)
        old_log_probs = tf.constant(nd_old_log_probs)
        advantages = tf.constant(nd_advantages)

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

        log_probs = tf.constant(nd_log_probs)
        old_log_probs = tf.constant(nd_old_log_probs)
        advantages = tf.constant(nd_advantages)

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

        log_probs = tf.constant(nd_log_probs)
        old_log_probs = tf.constant(nd_old_log_probs)
        advantages = tf.constant(nd_advantages)

        loss = build_policy_loss(log_probs, old_log_probs, advantages, 0.2)

        ratio = np.exp(nd_log_probs - nd_old_log_probs)
        ratio[ratio < 0.8] = 0.8
        answer = -np.mean(ratio * nd_advantages)

        with self.test_session() as sess:
            assert np.allclose(sess.run(loss), answer)

    def test_with_invalid_shape(self):
        epsilon = np.random.random()

        log_probs = tf.constant(np.random.random((4)))
        old_log_probs = tf.constant(np.random.random((4, 1)))
        advantages = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_policy_loss(log_probs, old_log_probs, advantages, epsilon)

        log_probs = tf.constant(np.random.random((4, 1)))
        old_log_probs = tf.constant(np.random.random((4)))
        advantages = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_policy_loss(log_probs, old_log_probs, advantages, epsilon)

        log_probs = tf.constant(np.random.random((4, 1)))
        old_log_probs = tf.constant(np.random.random((4, 1)))
        advantages = tf.constant(np.random.random((4)))
        with pytest.raises(AssertionError):
            build_policy_loss(log_probs, old_log_probs, advantages, epsilon)

        log_probs = tf.constant(np.random.random((4, 2)))
        old_log_probs = tf.constant(np.random.random((4, 1)))
        advantages = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_policy_loss(log_probs, old_log_probs, advantages, epsilon)

        log_probs = tf.constant(np.random.random((4, 1)))
        old_log_probs = tf.constant(np.random.random((4, 2)))
        advantages = tf.constant(np.random.random((4, 1)))
        with pytest.raises(AssertionError):
            build_policy_loss(log_probs, old_log_probs, advantages, epsilon)

        log_probs = tf.constant(np.random.random((4, 1)))
        old_log_probs = tf.constant(np.random.random((4, 1)))
        advantages = tf.constant(np.random.random((4, 2)))
        with pytest.raises(AssertionError):
            build_policy_loss(log_probs, old_log_probs, advantages, epsilon)

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
        self.network = PPONetwork(function(self.num_actions), self.state_shape,
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
            assert not np.all(var1 == var2)
