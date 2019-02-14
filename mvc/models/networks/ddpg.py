from collections import namedtuple

import tensorflow as tf
import numpy as np

from mvc.action_output import ActionOutput
from mvc.models.networks.base_network import BaseNetwork
from mvc.parametric_function import deterministic_policy_function
from mvc.parametric_function import q_function
from mvc.misc.assertion import assert_scalar


def initializer(shape, **kwargs):
    fan_in = int(shape[0])
    val = 1 / np.sqrt(fan_in)
    return tf.random_uniform(shape, -val, val, dtype=kwargs['dtype'])


def build_critic_loss(q_t, rewards_tp1, q_tp1, dones_tp1, gamma):
    assert_scalar(q_t)
    assert_scalar(rewards_tp1)
    assert_scalar(q_tp1)
    assert_scalar(dones_tp1)

    target = rewards_tp1 + gamma * q_tp1 * (1.0 - dones_tp1)
    loss = tf.reduce_mean(tf.square(target - q_t))
    return loss


def build_target_update(src, dst, tau):
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, src)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dst)
    ops = []
    for src_var, dst_var in zip(src_vars, dst_vars):
        ops.append(tf.assign(dst_var, dst_var * (1.0 - tau) + src_var * tau))
    return tf.group(*ops)


def build_optim(loss, learning_rate, scope):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    optimize_expr = optimizer.minimize(loss, var_list=variables)
    return optimize_expr


DDPGNetworkParams = namedtuple(
    'DDPGNetworkParams', ('fcs', 'concat_index', 'state_shape', 'num_actions',
                          'gamma', 'tau', 'actor_lr', 'critic_lr'))


class DDPGNetwork(BaseNetwork):
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

        # actor update
        actor_feed_dict = {
            self.obs_t_ph: kwargs['obs_t']
        }
        actor_ops = [self.actor_loss, self.actor_optimize_expr]
        actor_loss, _ = sess.run(actor_ops, feed_dict=actor_feed_dict)

        # target update
        sess.run([self.update_target_critic, self.update_target_actor])

        return critic_loss, actor_loss

    def _build(self, params):
        with tf.variable_scope('ddpg', reuse=tf.AUTO_REUSE):
            # placeholder
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

            last_initializer = tf.random_uniform_initializer(-3e-3, 3e-3)

            raw_policy_t = deterministic_policy_function(
                params.fcs, self.obs_t_ph, params.num_actions, tf.nn.tanh,
                w_init=initializer, last_w_init=last_initializer,
                last_b_init=last_initializer, scope='actor')
            policy_t = tf.nn.tanh(raw_policy_t)
            raw_policy_tp1 = deterministic_policy_function(
                params.fcs, self.obs_tp1_ph, params.num_actions, tf.nn.tanh,
                w_init=initializer, last_w_init=last_initializer,
                last_b_init=last_initializer, scope='target_actor')
            policy_tp1 = tf.nn.tanh(raw_policy_tp1)

            q_t = q_function(
                params.fcs, self.obs_t_ph, self.actions_t_ph,
                params.concat_index, tf.nn.tanh, w_init=initializer,
                last_w_init=last_initializer, last_b_init=last_initializer,
                scope='critic')
            q_t_with_actor = q_function(
                params.fcs, self.obs_t_ph, policy_t, params.concat_index,
                tf.nn.tanh, w_init=initializer, last_w_init=last_initializer,
                last_b_init=last_initializer, scope='critic')
            q_tp1 = q_function(
                params.fcs, self.obs_tp1_ph, policy_tp1, params.concat_index,
                tf.nn.tanh, w_init=initializer, last_w_init=last_initializer,
                last_b_init=last_initializer, scope='target_critic')

            # prepare for loss calculation
            rewards_tp1 = tf.reshape(self.rewards_tp1_ph, [-1, 1])
            dones_tp1 = tf.reshape(self.dones_tp1_ph, [-1, 1])

            # critic loss
            self.critic_loss = build_critic_loss(q_t, rewards_tp1, q_tp1,
                                                 dones_tp1, params.gamma)
            # actor loss
            self.actor_loss = -tf.reduce_mean(q_t_with_actor)

            # target update
            self.update_target_critic = build_target_update(
                'ddpg/critic', 'ddpg/target_critic', params.tau)
            self.update_target_actor = build_target_update(
                'ddpg/actor', 'ddpg/target_actor', params.tau)

            # optimization
            self.critic_optimize_expr = build_optim(
                self.critic_loss, params.critic_lr, 'ddpg/critic')
            self.actor_optimize_expr = build_optim(
                self.actor_loss, params.actor_lr, 'ddpg/actor')

            # action
            self.action = policy_t
            self.value = tf.reshape(q_t_with_actor, [-1])

    def _infer_arguments(self):
        return ['obs_t']

    def _update_arguments(self):
        return [
            'obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1'
        ]
