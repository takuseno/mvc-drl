from collections import namedtuple

import numpy as np
import tensorflow as tf

from mvc.models.networks.base_network import BaseNetwork
from mvc.action_output import ActionOutput
from mvc.parametric_function import stochastic_policy_function
from mvc.parametric_function import value_function
from mvc.misc.assertion import assert_scalar


def build_value_loss(values, returns, old_values, epsilon, value_factor):
    assert_scalar(values)
    assert_scalar(returns)
    assert_scalar(old_values)

    with tf.variable_scope('value_loss'):
        clipped_diff = tf.clip_by_value(
            (values - old_values), -epsilon, epsilon)
        loss_clipped = (old_values + clipped_diff - returns) ** 2
        loss_non_clipped = (returns - values) ** 2
        loss = tf.reduce_mean(tf.maximum(loss_clipped, loss_non_clipped))
        return value_factor * loss


def build_entropy_loss(dist, entropy_factor):
    with tf.variable_scope('entropy'):
        entropy = -tf.reduce_mean(dist.entropy())
        return entropy_factor * entropy


def build_policy_loss(log_probs, old_log_probs, advantages, epsilon):
    assert_scalar(log_probs)
    assert_scalar(old_log_probs)
    assert_scalar(advantages)

    with tf.variable_scope('policy_loss'):
        ratio = tf.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(
            ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        surr = tf.minimum(surr1, surr2)
        loss = -tf.reduce_mean(surr)
    return loss


PPONetworkParams = namedtuple(
    'PPONetworkParams', ('fcs', 'state_shape', 'num_envs', 'num_actions',
                         'batch_size', 'epsilon', 'learning_rate', 'grad_clip',
                         'value_factor', 'entropy_factor'))


class PPONetwork(BaseNetwork):
    def __init__(self, params):

        self._build(params)

    def _infer(self, **kwargs):
        feed_dict = {
            self.step_obs_ph: kwargs['obs_t'],
        }
        sess = tf.get_default_session()
        ops = [self.action, self.log_policy, self.value]
        return ActionOutput(*sess.run(ops, feed_dict=feed_dict))

    def _update(self, **kwargs):
        feed_dict = {
            self.train_obs_ph: kwargs['obs_t'],
            self.actions_ph: kwargs['actions_t'],
            self.returns_ph: kwargs['returns_t'],
            self.advantages_ph: kwargs['advantages_t'],
            self.old_log_probs_ph: kwargs['log_probs_t'],
            self.old_values_ph: kwargs['values_t']
        }
        sess = tf.get_default_session()
        opts = [self.loss, self.optimize_expr]
        return sess.run(opts, feed_dict=feed_dict)[0]

    def _build(self, params):
        with tf.variable_scope('ppo', reuse=tf.AUTO_REUSE):
            # placeholers
            self.step_obs_ph = tf.placeholder(
                tf.float32, [params.num_envs] + list(params.state_shape),
                name='step_obs')
            self.train_obs_ph = tf.placeholder(
                tf.float32, [params.batch_size] + list(params.state_shape),
                name='train_obs')
            self.returns_ph = tf.placeholder(
                tf.float32, [params.batch_size], name='returns')
            self.advantages_ph = tf.placeholder(
                tf.float32, [params.batch_size], name='advantages')
            self.actions_ph = tf.placeholder(
                tf.float32, [params.batch_size, params.num_actions],
                name='action')
            self.old_log_probs_ph = tf.placeholder(
                tf.float32, [params.batch_size], name='old_log_prob')
            self.old_values_ph = tf.placeholder(
                tf.float32, [params.batch_size], 'old_values')

            def initializer(scale):
                input_dim = int(params.state_shape[0])
                stddev = np.sqrt(scale / input_dim)
                return tf.random_normal_initializer(stddev=stddev)

            # network outputs for inference
            step_dist = stochastic_policy_function(
                params.fcs, self.step_obs_ph, params.num_actions, tf.nn.tanh,
                w_init=initializer(1.0), last_w_init=initializer(0.01),
                scope='pi')
            step_values = value_function(
                params.fcs, self.step_obs_ph, tf.nn.tanh, initializer(1.0),
                initializer(1.0), scope='v')

            # network outputs for training
            train_dist = stochastic_policy_function(
                params.fcs, self.train_obs_ph, params.num_actions, tf.nn.tanh,
                w_init=initializer(1.0), last_w_init=initializer(0.01),
                scope='pi')
            train_values = value_function(
                params.fcs, self.train_obs_ph, tf.nn.tanh, initializer(1.0),
                initializer(1.0), scope='v')

            # individual loss
            value_loss = build_value_loss(
                train_values, tf.reshape(self.returns_ph, [-1, 1]),
                tf.reshape(self.old_values_ph, [-1, 1]), params.epsilon,
                params.value_factor)
            entropy_loss = build_entropy_loss(
                train_dist, params.entropy_factor)
            policy_loss = build_policy_loss(
                tf.reshape(train_dist.log_prob(self.actions_ph), [-1, 1]),
                tf.reshape(self.old_log_probs_ph, [-1, 1]),
                tf.reshape(self.advantages_ph, [-1, 1]), params.epsilon)
            # final loss
            self.loss = value_loss + policy_loss + entropy_loss

            # network weights
            network_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'ppo')

            # gradients
            gradients, _ = tf.clip_by_global_norm(
                tf.gradients(self.loss, network_vars), params.grad_clip)

            # update
            optimizer = tf.train.AdamOptimizer(
                params.learning_rate, epsilon=1e-5)
            self.optimize_expr = optimizer.apply_gradients(
                zip(gradients, network_vars))

            # action
            self.action = step_dist.sample(1)[0]
            self.log_policy = tf.reshape(step_dist.log_prob(self.action), [-1])
            self.value = tf.reshape(step_values, [-1])

    def _infer_arguments(self):
        return ['obs_t']

    def _update_arguments(self):
        return [
            'obs_t', 'actions_t', 'log_probs_t', 'returns_t',
            'advantages_t', 'values_t'
        ]
