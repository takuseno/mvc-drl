import tensorflow as tf
import numpy as np
import unittest
import pytest

from tests.test_utils import assert_variable_mismatch, assert_variable_match
from tests.test_utils import make_fcs, to_tf

from mvc.models.networks.td3 import build_target
from mvc.models.networks.td3 import build_critic_loss
from mvc.models.networks.td3 import build_actor_loss
from mvc.models.networks.td3 import TD3NetworkParams
from mvc.models.networks.td3 import TD3Network


class BuildTarget(tf.test.TestCase):
    def test_success(self):
        nd_rewards_tp1 = np.random.random((4, 1))
        nd_q1_tp1 = np.random.random((4, 1))
        nd_q2_tp1 = np.random.random((4, 1))
        nd_dones_tp1 = np.random.randint(2, size=(4, 1))
        gamma = np.random.random()

        rewards_tp1 = to_tf(nd_rewards_tp1)
        q1_tp1 = to_tf(nd_q1_tp1)
        q2_tp1 = to_tf(nd_q2_tp1)
        dones_tp1 = to_tf(nd_dones_tp1)

        target = build_target(rewards_tp1, q1_tp1, q2_tp1, dones_tp1, gamma)

        with self.test_session() as sess:
            q_tp1 = np.reshape(np.min(np.hstack([nd_q1_tp1, nd_q2_tp1]), axis=1), (-1, 1))
            nd_target = nd_rewards_tp1 + gamma * q_tp1 * (1.0 - nd_dones_tp1)
            assert np.allclose(sess.run(target), nd_target)


class BuildCriticLoss(tf.test.TestCase):
    def test_success(self):
        nd_q1_t = np.random.random((4, 1))
        nd_q2_t = np.random.random((4, 1))
        nd_target = np.random.random((4, 1))

        q1_t = to_tf(nd_q1_t)
        q2_t = to_tf(nd_q2_t)
        target = to_tf(nd_target)

        loss = build_critic_loss(q1_t, q2_t, target)

        with self.test_session() as sess:
            nd_loss = np.mean((nd_target - nd_q1_t) ** 2) + np.mean((nd_target - nd_q2_t) ** 2)
            assert np.allclose(sess.run(loss), nd_loss)


class BuildActorLoss(tf.test.TestCase):
    def test_success(self):
        nd_q1_t = np.random.random((4, 1))
        nd_q2_t = np.random.random((4, 1))

        q1_t = to_tf(nd_q1_t)
        q2_t = to_tf(nd_q2_t)

        loss = build_actor_loss(q1_t, q2_t)

        with self.test_session() as sess:
            nd_loss = np.mean(np.min(np.hstack([nd_q1_t, nd_q2_t]), axis=1))
            assert np.allclose(sess.run(loss), nd_loss)


class TD3NetworkTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        fcs = make_fcs()
        concat_index = np.random.randint(len(fcs))
        state_shape = (np.random.randint(5) + 1,)
        num_actions = np.random.randint(5) + 1
        gamma = np.random.random()
        tau = np.random.random()
        actor_lr = np.random.random()
        critic_lr = np.random.random()
        target_noise_sigma = np.random.random()
        target_noise_clip = np.random.random()

        self.params = TD3NetworkParams(fcs=fcs, concat_index=concat_index,
                                       state_shape=state_shape,
                                       num_actions=num_actions,
                                       gamma=gamma, tau=tau, actor_lr=actor_lr,
                                       critic_lr=critic_lr,
                                       target_noise_sigma=target_noise_sigma,
                                       target_noise_clip=target_noise_clip)
        self.network = TD3Network(self.params)

    def test_build(self):
        assert int(self.network.action.shape[1]) == self.params.num_actions
        assert len(self.network.value.shape) == 1
        assert len(self.network.actor_loss.shape) == 0
        assert len(self.network.critic_loss.shape) == 0

    def test_infer_arguments(self):
        args = self.network._infer_arguments()
        keys = ['obs_t']
        for key in args:
            assert key in keys

    def test_infer(self):
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            obs = np.random.random(self.params.state_shape)
            output = self.network.infer(obs_t=obs)

        assert output.action.shape == (self.params.num_actions,)
        assert output.log_prob is None
        assert len(output.value.shape) == 0

    def test_update_only_critic(self):
        obs_t = np.random.random((32,) + self.params.state_shape)
        actions_t = np.random.random((32, self.params.num_actions))
        rewards_tp1 = np.random.random((32,))
        obs_tp1 = np.random.random((32,) + self.params.state_shape)
        dones_tp1 = np.random.random((32,))
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'td3/critic')
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'td3/actor')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            before_critic = sess.run(critic_variables)
            before_actor = sess.run(actor_variables)

            critic_loss, actor_loss = self.network.update(
                obs_t=obs_t, actions_t=actions_t, rewards_tp1=rewards_tp1,
                obs_tp1=obs_tp1, dones_tp1=dones_tp1, update_actor=False)

            after_critic = sess.run(critic_variables)
            after_actor = sess.run(actor_variables)

        assert_variable_mismatch(before_critic, after_critic)
        assert_variable_match(before_actor, after_actor)
        assert actor_loss is None

    def test_update_both(self):
        obs_t = np.random.random((32,) + self.params.state_shape)
        actions_t = np.random.random((32, self.params.num_actions))
        rewards_tp1 = np.random.random((32,))
        obs_tp1 = np.random.random((32,) + self.params.state_shape)
        dones_tp1 = np.random.random((32,))
        critic_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'td3/critic')
        actor_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'td3/actor')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            before_critic = sess.run(critic_variables)
            before_actor = sess.run(actor_variables)

            critic_loss, actor_loss = self.network.update(
                obs_t=obs_t, actions_t=actions_t, rewards_tp1=rewards_tp1,
                obs_tp1=obs_tp1, dones_tp1=dones_tp1, update_actor=True)

            after_critic = sess.run(critic_variables)
            after_actor = sess.run(actor_variables)

        assert_variable_mismatch(before_critic, after_critic)
        assert_variable_mismatch(before_actor, after_actor)
        assert actor_loss is not None
