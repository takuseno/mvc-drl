from collections import namedtuple

import tensorflow as tf
import numpy as np

from mvc.action_output import ActionOutput
from mvc.models.networks.base_network import BaseNetwork
from mvc.models.networks.ddpg import initializer, build_target_update
from mvc.models.networks.ddpg import build_optim
from mvc.parametric_function import deterministic_policy_function
from mvc.parametric_function import q_function
from mvc.misc.assertion import assert_scalar


def build_smoothed_target(policy_tp1, sigma, c):
    smoothing_noise = tf.random.normal(policy_tp1.shape[1:], 0.0, sigma)
    clipped_noise = tf.clip_by_value(smoothing_noise, -c, c)
    return tf.clip_by_value(policy_tp1 + clipped_noise, -1.0, 1.0)


def build_target(rewards_tp1, q1_tp1, q2_tp1, dones_tp1, gamma):
    assert_scalar(rewards_tp1)
    assert_scalar(q1_tp1)
    assert_scalar(q2_tp1)
    assert_scalar(dones_tp1)

    q_tp1 = tf.minimum(q1_tp1, q2_tp1)
    target = rewards_tp1 + gamma * q_tp1 * (1.0 - dones_tp1)
    return tf.stop_gradient(target)


def build_critic_loss(q1_t, q2_t, target):
    q1_loss = tf.reduce_mean(tf.square(target - q1_t))
    q2_loss = tf.reduce_mean(tf.square(target - q2_t))
    return q1_loss + q2_loss


def build_actor_loss(q1_t, q2_t):
    assert_scalar(q1_t)
    assert_scalar(q2_t)

    q_t = tf.minimum(q1_t, q2_t)
    loss = tf.reduce_mean(q_t)
    return loss


TD3NetworkParams = namedtuple(
    'TD3NetworkParams', ('fcs', 'concat_index', 'state_shape', 'num_actions',
                         'gamma', 'tau', 'actor_lr', 'critic_lr',
                         'target_noise_sigma', 'target_noise_clip'))


last_initializer = tf.random_uniform_initializer(-3e-3, 3e-3)


def _q_function(params, obs, action, scope):
    return q_function(params.fcs, obs, action, params.concat_index,
                      tf.nn.tanh, w_init=initializer,
                      last_w_init=last_initializer,
                      last_b_init=last_initializer, scope=scope)


def _policy_function(params, obs, scope):
    return deterministic_policy_function(
        params.fcs, obs, params.num_actions, tf.nn.tanh, w_init=initializer,
        last_w_init=last_initializer, last_b_init=last_initializer,
        scope=scope)


class TD3Network(BaseNetwork):
    def __init__(self, params):
        self._build(params)

    def _infer(self, **kwargs):
        feed_dict = {
            self.obs_t_ph: np.array([kwargs['obs_t']])
        }
        sess = tf.get_default_session()
        ops = [self.action, self.value]
        action, value = sess.run(ops, feed_dict=feed_dict)
        return ActionOutput(action=action[0], log_prob=None, value=value[0])

    def _update(self, **kwargs):
        sess = tf.get_default_session()

        # critic update
        critic_feed_dict = {
            self.obs_t_ph: kwargs['obs_t'],
            self.actions_t_ph: kwargs['actions_t'],
            self.rewards_tp1_ph: kwargs['rewards_tp1'],
            self.obs_tp1_ph: kwargs['obs_tp1'],
            self.dones_tp1_ph: kwargs['dones_tp1']
        }
        critic_ops = [self.critic_loss, self.critic_optimize_expr]
        critic_loss, _ = sess.run(critic_ops, feed_dict=critic_feed_dict)

        # actor update (delayed policy update)
        if kwargs['update_actor']:
            actor_feed_dict = {
                self.obs_t_ph: kwargs['obs_t']
            }
            actor_ops = [self.actor_loss, self.actor_optimize_expr]
            actor_loss, _ = sess.run(actor_ops, feed_dict=actor_feed_dict)

            # target update
            sess.run([self.update_target_critic, self.update_target_actor])
        else:
            actor_loss = None

        return critic_loss, actor_loss

    def _build(self, params):
        with tf.variable_scope('td3', reuse=tf.AUTO_REUSE):
            self.obs_t_ph = tf.placeholder(
                tf.float32, [None] + list(params.state_shape), name='obs_t')
            self.actions_t_ph = tf.placeholder(
                tf.float32, [None, params.num_actions], name='actions_t')
            self.rewards_tp1_ph = tf.placeholder(
                tf.float32, [None], name='rewards_tp1')
            self.obs_tp1_ph = tf.placeholder(
                tf.float32, [None] + list(params.state_shape), name='obs_tp1')
            self.dones_tp1_ph = tf.placeholder(
                tf.float32, [None], name='dones_tp1')

            # policy function
            raw_policy_t = _policy_function(params, self.obs_t_ph, 'actor')
            policy_t = tf.nn.tanh(raw_policy_t)

            # target policy function
            raw_policy_tp1 = _policy_function(params, self.obs_tp1_ph,
                                              'target_actor')
            policy_tp1 = tf.nn.tanh(raw_policy_tp1)

            # target policy smoothing reguralization
            smoothed_policy_tp1 = build_smoothed_target(
                policy_tp1, params.target_noise_sigma,
                params.target_noise_clip)

            # first critic
            q1_t = _q_function(
                params, self.obs_t_ph, self.actions_t_ph, 'critic/1')
            q1_t_with_actor = _q_function(
                params, self.obs_t_ph, policy_t, 'critic/1')

            # first target critic
            q1_tp1 = _q_function(params, self.obs_tp1_ph, smoothed_policy_tp1,
                                 'target_critic/1')

            # second critic
            q2_t = _q_function(
                params, self.obs_t_ph, self.actions_t_ph, 'critic/2')
            q2_t_with_actor = _q_function(
                params, self.obs_t_ph, policy_t, 'critic/2')

            # second target critic
            q2_tp1 = _q_function(params, self.obs_tp1_ph, smoothed_policy_tp1,
                                 'target_critic/2')

            # prepare for loss calculation
            rewards_tp1 = tf.reshape(self.rewards_tp1_ph, [-1, 1])
            dones_tp1 = tf.reshape(self.dones_tp1_ph, [-1, 1])

            # critic loss
            target = build_target(
                rewards_tp1, q1_tp1, q2_tp1, dones_tp1, params.gamma)
            self.critic_loss = build_critic_loss(q1_t, q2_t, target)

            # actor loss
            self.actor_loss = -build_actor_loss(
                q1_t_with_actor, q2_t_with_actor)

            # target update
            self.update_target_critic = build_target_update(
                'td3/critic', 'td3/target_critic', params.tau)
            self.update_target_actor = build_target_update(
                'td3/actor', 'td3/target_actor', params.tau)

            # optimization
            self.critic_optimize_expr = build_optim(
                self.critic_loss, params.critic_lr, 'td3/critic')
            self.actor_optimize_expr = build_optim(
                self.actor_loss, params.actor_lr, 'td3/actor')

            # action
            self.action = policy_t
            self.value = tf.reshape(q1_t_with_actor, [-1])

    def _infer_arguments(self):
        return ['obs_t']

    def _update_arguments(self):
        return [
            'obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1',
            'update_actor'
        ]
