from collections import namedtuple

import tensorflow as tf
import numpy as np

from mvc.action_output import ActionOutput
from mvc.models.networks.base_network import BaseNetwork
from mvc.parametric_function import stochastic_policy_function
from mvc.parametric_function import q_function, value_function
from mvc.models.networks.ddpg import build_target_update
from mvc.models.networks.ddpg import build_optim
from mvc.misc.assertion import assert_scalar


def build_v_loss(v_t, q1_t, q2_t, log_prob_t):
    assert_scalar(v_t)
    assert_scalar(q1_t)
    assert_scalar(q2_t)
    assert_scalar(log_prob_t)

    q_t = tf.minimum(q1_t, q2_t)
    target = tf.stop_gradient(q_t - log_prob_t)
    loss = 0.5 * tf.reduce_mean((v_t - target) ** 2)
    return loss


def build_q_loss(q_t, rewards_tp1, v_tp1, dones_tp1, gamma):
    assert_scalar(q_t)
    assert_scalar(rewards_tp1)
    assert_scalar(v_tp1)
    assert_scalar(dones_tp1)

    target = tf.stop_gradient(rewards_tp1 + gamma * v_tp1 * (1.0 - dones_tp1))
    loss = 0.5 * tf.reduce_mean((target - q_t) ** 2)
    return loss


def build_pi_loss(log_prob_t, q1_t, q2_t):
    assert_scalar(log_prob_t)
    assert_scalar(q1_t)
    assert_scalar(q2_t)

    q_t = tf.minimum(q1_t, q2_t)
    loss = tf.reduce_mean(log_prob_t - q_t)
    return loss


def build_weight_decay(scale, scope):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    weight_sum = 0.0
    for var in variables:
        if var.name.find('bias') > -1:
            continue
        weight_sum += tf.reduce_sum(var)
    return scale * weight_sum


def squash_action(dist):
    sampled_action = dist.sample(1)[0]
    squashed_action = tf.nn.tanh(sampled_action)
    diff = tf.reduce_sum(
        tf.log(1 - squashed_action ** 2 + 1e-6), axis=1, keepdims=True)
    log_prob = tf.reshape(dist.log_prob(sampled_action), [-1, 1]) - diff
    return squashed_action, log_prob


def build_policy_reg(dist, scale):
    mean_loss = 0.5 * tf.reduce_mean(dist.mean() ** 2)
    logstd_loss = 0.5 * tf.reduce_mean(tf.log(dist.stddev()) ** 2)
    return scale * (mean_loss + logstd_loss)


SACNetworkParams = namedtuple(
    'SACNetworkParams', ('fcs', 'concat_index', 'state_shape', 'num_actions',
                         'gamma', 'tau', 'pi_lr', 'q_lr', 'v_lr', 'reg'))


# initialzier
ZEROS_INIT = tf.zeros_initializer()
XAVIER_INIT = tf.contrib.layers.xavier_initializer()


class SACNetwork(BaseNetwork):
    def __init__(self, params):
        self._build(params)

    def _infer(self, **kwargs):
        sess = tf.get_default_session()
        feed_dict = {
            self.obs_t_ph: np.array([kwargs['obs_t']])
        }
        ops = [self.action, self.log_prob, self.value]
        return ActionOutput(*sess.run(ops, feed_dict=feed_dict))

    def _update(self, **kwargs):
        sess = tf.get_default_session()

        # update value function
        v_feed_dict = {
            self.obs_t_ph: kwargs['obs_t']
        }
        v_ops = [self.v_loss, self.v_optimize_expr]
        v_loss, _ = sess.run(v_ops, feed_dict=v_feed_dict)

        # update q functions
        q_feed_dict = {
            self.obs_t_ph: kwargs['obs_t'],
            self.actions_t_ph: kwargs['actions_t'],
            self.rewards_tp1_ph: kwargs['rewards_tp1'],
            self.obs_tp1_ph: kwargs['obs_tp1'],
            self.dones_tp1_ph: kwargs['dones_tp1']
        }
        q_ops = [
            self.q1_loss, self.q2_loss,
            self.q1_optimize_expr, self.q2_optimize_expr
        ]
        q1_loss, q2_loss, _, _ = sess.run(q_ops, feed_dict=q_feed_dict)

        # update policy function
        pi_feed_dict = {
            self.obs_t_ph: kwargs['obs_t']
        }
        pi_ops = [self.pi_loss, self.pi_optimize_expr]
        pi_loss, _ = sess.run(pi_ops, feed_dict=pi_feed_dict)

        # update target function
        sess.run(self.target_update)

        return v_loss, (q1_loss, q2_loss), pi_loss

    def _build(self, params):
        with tf.variable_scope('sac'):
            self.obs_t_ph = tf.placeholder(
                tf.float32, (None,) + params.state_shape, name='obs_t')
            self.actions_t_ph = tf.placeholder(
                tf.float32, (None, params.num_actions), name='actions_t')
            self.rewards_tp1_ph = tf.placeholder(
                tf.float32, (None,), name='rewards_tp1')
            self.obs_tp1_ph = tf.placeholder(
                tf.float32, (None,) + params.state_shape, name='obs_tp1')
            self.dones_tp1_ph = tf.placeholder(
                tf.float32, (None,), name='dones_tp1')

            # policy function
            pi_t = stochastic_policy_function(params.fcs, self.obs_t_ph,
                                              params.num_actions,
                                              tf.nn.relu, share=True,
                                              w_init=XAVIER_INIT,
                                              last_w_init=XAVIER_INIT,
                                              last_b_init=XAVIER_INIT,
                                              scope='pi')
            squashed_action_t, log_prob_t = squash_action(pi_t)

            # value function
            v_t = value_function(
                params.fcs, self.obs_t_ph, tf.nn.relu, XAVIER_INIT,
                XAVIER_INIT, ZEROS_INIT, scope='v')
            # target value function
            v_tp1 = value_function(
                params.fcs, self.obs_tp1_ph, tf.nn.relu, XAVIER_INIT,
                XAVIER_INIT, ZEROS_INIT, scope='target_v')

            # two q functions
            q1_t_with_pi = q_function(params.fcs, self.obs_t_ph,
                                      squashed_action_t, params.concat_index,
                                      tf.nn.relu, XAVIER_INIT,
                                      XAVIER_INIT, ZEROS_INIT, scope='q1')
            q1_t = q_function(params.fcs, self.obs_t_ph, self.actions_t_ph,
                              params.concat_index, tf.nn.relu, XAVIER_INIT,
                              XAVIER_INIT, ZEROS_INIT, scope='q1')
            q2_t_with_pi = q_function(params.fcs, self.obs_t_ph,
                                      squashed_action_t, params.concat_index,
                                      tf.nn.relu, XAVIER_INIT,
                                      XAVIER_INIT, ZEROS_INIT, scope='q2')
            q2_t = q_function(params.fcs, self.obs_t_ph, self.actions_t_ph,
                              params.concat_index, tf.nn.relu, XAVIER_INIT,
                              XAVIER_INIT, ZEROS_INIT, scope='q2')

            # prepare for loss
            rewards_tp1 = tf.reshape(self.rewards_tp1_ph, [-1, 1])
            dones_tp1 = tf.reshape(self.dones_tp1_ph, [-1, 1])

            # value function loss
            self.v_loss = build_v_loss(
                v_t, q1_t_with_pi, q2_t_with_pi, log_prob_t)
            # q function loss
            self.q1_loss = build_q_loss(
                q1_t, rewards_tp1, v_tp1, dones_tp1, params.gamma)
            self.q2_loss = build_q_loss(
                q2_t, rewards_tp1, v_tp1, dones_tp1, params.gamma)
            # policy function loss
            self.pi_loss = build_pi_loss(
                log_prob_t, q1_t_with_pi, q2_t_with_pi)
            # policy reguralization
            policy_decay = build_policy_reg(pi_t, params.reg)

            # target update
            self.target_update = build_target_update(
                'sac/v', 'sac/target_v', params.tau)

            # optimization
            self.v_optimize_expr = build_optim(
                self.v_loss, params.v_lr, 'sac/v')
            self.q1_optimize_expr = build_optim(
                self.q1_loss, params.q_lr, 'sac/q1')
            self.q2_optimize_expr = build_optim(
                self.q2_loss, params.q_lr, 'sac/q2')
            self.pi_optimize_expr = build_optim(
                self.pi_loss + policy_decay, params.pi_lr, 'sac/pi')

            # for inference
            self.action = squashed_action_t[0]
            self.value = tf.reshape(v_t, [-1])[0]
            self.log_prob = tf.reshape(log_prob_t, [-1])[0]

    def _infer_arguments(self):
        return ['obs_t']

    def _update_arguments(self):
        return ['obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1']
