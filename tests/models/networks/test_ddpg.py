import tensorflow as tf
import numpy as np
import unittest
import pytest

from tests.test_utils import assert_variable_mismatch, assert_variable_match
from tests.test_utils import make_fcs
from mvc.models.networks.ddpg import build_critic_loss
from mvc.models.networks.ddpg import build_target_update
from mvc.models.networks.ddpg import build_optimization
from mvc.models.networks.ddpg import DDPGNetwork


class BuildCriticLoss(tf.test.TestCase):
    def test_success(self):
        nd_q_t = np.random.random((4, 1))
        nd_rewards_tp1 = np.random.random((4, 1))
        nd_q_tp1 = np.random.random((4, 1))
        nd_dones_tp1 = np.random.randint(2, size=(4, 1))
        q_t = tf.constant(nd_q_t, dtype=tf.float32)
        rewards_tp1 = tf.constant(nd_rewards_tp1, dtype=tf.float32)
        q_tp1 = tf.constant(nd_q_tp1, dtype=tf.float32)
        dones_tp1 = tf.constant(nd_dones_tp1, dtype=tf.float32)
        gamma = np.random.random()

        loss = build_critic_loss(q_t, rewards_tp1, q_tp1, dones_tp1, gamma)

        with self.test_session() as sess:
            target = nd_rewards_tp1 + gamma * nd_q_tp1 * (1.0 - nd_dones_tp1)
            answer = np.mean((target - nd_q_t) ** 2)
            assert np.allclose(sess.run(loss), answer)


class BuildTargetUpdateTest(tf.test.TestCase):
    def test_success(self):
        dim1 = np.random.randint(10) + 1
        dim2 = np.random.randint(10) + 1
        tau = np.random.random()
        var1 = tf.Variable(np.random.random((dim1, dim2)), name='var1')
        var2 = tf.Variable(np.random.random((dim1, dim2)), name='var2')

        ops = build_target_update('var1', 'var2', tau)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            before_var1, before_var2 = sess.run([var1, var2])
            sess.run(ops)
            after_var1, after_var2 = sess.run([var1, var2])

            assert_variable_match(before_var1, after_var1)

            assert np.allclose((1.0 - tau) * before_var2 + tau * before_var1, after_var2)


class BuildOptimization(tf.test.TestCase):
    def test_success(self):
        dim1 = np.random.randint(10) + 1
        dim2 = np.random.randint(10) + 1
        var1 = tf.Variable(np.random.random((dim1, dim2)), name='var1')
        var2 = tf.Variable(np.random.random((dim1, dim2)), name='var2')

        ops = build_optimization(var1, 1e-4, 'var1')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            before_var1, before_var2 = sess.run([var1, var2])
            sess.run(ops)
            after_var1, after_var2 = sess.run([var1, var2])

            assert_variable_mismatch(before_var1, after_var1)
            assert_variable_match(before_var2, after_var2)


class DDPGNetworkTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.fcs = make_fcs()
        self.concat_index = np.random.randint(len(self.fcs))
        self.state_shape = (np.random.randint(5) + 1,)
        self.num_actions = np.random.randint(5) + 1
        self.gamma = np.random.random()
        self.tau = np.random.random()
        self.actor_lr = np.random.random()
        self.critic_lr = np.random.random()
        self.network = DDPGNetwork(self.fcs, self.concat_index, self.state_shape,
                                   self.num_actions, self.gamma, self.tau,
                                   self.actor_lr, self.critic_lr)

    def test_build(self):
        assert int(self.network.action.shape[1]) == self.num_actions
        assert len(self.network.value.shape) == 1
        assert len(self.network.actor_loss.shape) == 0
        assert len(self.network.critic_loss.shape) == 0

    def test_infer_arguments(self):
        args = self.network._infer_arguments()
        keys = ['obs_t']
        for key in args:
            assert key in keys

    def test_update_arguments(self):
        args = self.network._update_arguments()
        keys = ['obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1']
        for key in args:
            assert key in keys

    def test_infer(self):
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            obs = np.random.random((1,) + self.state_shape)
            output = self.network.infer(obs_t=obs)

        assert output.action.shape == (1, self.num_actions)
        assert output.log_prob is None
        assert output.value.shape == (1,)

    def test_update(self):
        obs_t = np.random.random((32,) + self.state_shape)
        actions_t = np.random.random((32, self.num_actions))
        rewards_tp1 = np.random.random((32,))
        obs_tp1 = np.random.random((32,) + self.state_shape)
        dones_tp1 = np.random.random((32,))
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ddpg')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(variables)

            critic_loss, actor_loss = self.network.update(
                obs_t=obs_t, actions_t=actions_t, rewards_tp1=rewards_tp1,
                obs_tp1=obs_tp1, dones_tp1=dones_tp1)

            after = sess.run(variables)

        assert_variable_mismatch(before, after)
