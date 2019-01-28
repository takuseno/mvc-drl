import tensorflow as tf
import numpy as np

from mvc.action_output import ActionOutput
from mvc.models.networks.base_network import BaseNetwork
from mvc.parametric_function import stochastic_policy_function
from mvc.parametric_function import q_function, value_function
from mvc.models.networks.ddpg import initializer
from mvc.models.networks.ddpg import build_target_update
from mvc.models.networks.ddpg import build_optimization


def build_v_loss(v_t, q1_t, q2_t, log_prob_t):
    assert len(v_t.shape) == 2 and v_t.shape[1] == 1
    assert len(q1_t.shape) == 2 and q1_t.shape[1] == 1
    assert len(q2_t.shape) == 2 and q2_t.shape[1] == 1
    assert len(log_prob_t.shape) == 2

    q_t = tf.minimum(q1_t, q2_t)
    target = tf.stop_gradient(q_t - log_prob_t)
    loss = 0.5 * tf.reduce_mean((v_t - target) ** 2)
    return loss


def build_q_loss(q_t, rewards_tp1, v_tp1, dones_tp1, gamma):
    assert len(q_t.shape) == 2 and q_t.shape[1] == 1
    assert len(rewards_tp1.shape) == 2 and rewards_tp1.shape[1] == 1
    assert len(v_tp1.shape) == 2 and v_tp1.shape[1] == 1
    assert len(dones_tp1.shape) == 2 and dones_tp1.shape[1] == 1

    target = tf.stop_gradient(rewards_tp1 + gamma * v_tp1 * (1.0 - dones_tp1))
    loss = 0.5 * tf.reduce_mean((target - q_t) ** 2)
    return loss


def build_pi_loss(log_prob_t, q1_t, q2_t):
    assert len(log_prob_t.shape) == 2
    assert len(q1_t.shape) == 2 and q1_t.shape[1] == 1
    assert len(q2_t.shape) == 2 and q2_t.shape[1] == 1

    q_t = tf.stop_gradient(tf.minimum(q1_t, q2_t))
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


class SACNetwork(BaseNetwork):
    def __init__(self,
                 fcs,
                 concat_index,
                 state_shape,
                 num_actions,
                 gamma,
                 tau,
                 pi_lr,
                 q_lr,
                 v_lr):
        self._build(fcs, concat_index, state_shape, num_actions,
                    gamma, tau, pi_lr, q_lr, v_lr)

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

    def _build(self,
               fcs,
               concat_index,
               state_shape,
               num_actions,
               gamma,
               tau,
               pi_lr,
               q_lr,
               v_lr):
        with tf.variable_scope('sac'):
            obs_t_ph = self.obs_t_ph = tf.placeholder(
                tf.float32, (None,) + state_shape, name='obs_t')
            actions_t_ph = self.actions_t_ph = tf.placeholder(
                tf.float32, (None, num_actions), name='actions_t')
            rewards_tp1_ph = self.rewards_tp1_ph = tf.placeholder(
                tf.float32, (None,), name='rewards_tp1')
            obs_tp1_ph = self.obs_tp1_ph = tf.placeholder(
                tf.float32, (None,) + state_shape, name='obs_tp1')
            dones_tp1_ph = self.dones_tp1_ph = tf.placeholder(
                tf.float32, (None,), name='dones_tp1')

            # initialzier
            last_w_init = tf.contrib.layers.xavier_initializer()
            last_b_init = tf.constant_initializer(0.0)

            # policy function
            pi_t = stochastic_policy_function(fcs, obs_t_ph, num_actions,
                                              tf.nn.relu, share=True,
                                              w_init=initializer,
                                              last_w_init=last_w_init,
                                              last_b_init=last_b_init,
                                              scope='pi')
            sampled_action_t = tf.stop_gradient(pi_t.sample(1)[0])
            log_prob_t = pi_t.log_prob(sampled_action_t)
            squashed_action_t = tf.nn.tanh(sampled_action_t)

            # value function
            v_t = value_function(
                fcs, obs_t_ph, tf.nn.relu, initializer,
                last_w_init, last_b_init, scope='v')
            # target value function
            v_tp1 = value_function(
                fcs, obs_tp1_ph, tf.nn.relu, initializer,
                last_w_init, last_b_init, scope='target_v')

            # two q functions
            q1_t_with_pi = q_function(fcs, obs_t_ph, squashed_action_t,
                                      concat_index, tf.nn.relu, initializer,
                                      last_w_init, last_b_init, scope='q1')
            q1_t = q_function(fcs, obs_t_ph, actions_t_ph, concat_index,
                              tf.nn.relu, initializer, last_w_init,
                              last_b_init, scope='q1')
            q2_t_with_pi = q_function(fcs, obs_t_ph, squashed_action_t,
                                      concat_index, tf.nn.relu, initializer,
                                      last_w_init, last_b_init, scope='q2')
            q2_t = q_function(fcs, obs_t_ph, actions_t_ph, concat_index,
                              tf.nn.relu, initializer, last_w_init,
                              last_b_init, scope='q2')

            # prepare for loss
            rewards_tp1 = tf.reshape(rewards_tp1_ph, [-1, 1])
            dones_tp1 = tf.reshape(dones_tp1_ph, [-1, 1])

            # value function loss
            self.v_loss = build_v_loss(
                v_t, q1_t_with_pi, q2_t_with_pi, log_prob_t)
            # q function loss
            self.q1_loss = build_q_loss(
                q1_t, rewards_tp1, v_tp1, dones_tp1, gamma)
            self.q2_loss = build_q_loss(
                q2_t, rewards_tp1, v_tp1, dones_tp1, gamma)
            # policy function loss
            self.pi_loss = build_pi_loss(
                log_prob_t, q1_t_with_pi, q2_t_with_pi)

            # target update
            self.target_update = build_target_update(
                'sac/v', 'sac/target_v', tau)

            # policy weight decay
            policy_decay = build_weight_decay(0.001, 'sac/pi')

            # optimization
            self.v_optimize_expr = build_optimization(
                self.v_loss, v_lr, 'sac/v')
            self.q1_optimize_expr = build_optimization(
                self.q1_loss, q_lr, 'sac/q1')
            self.q2_optimize_expr = build_optimization(
                self.q2_loss, q_lr, 'sac/q2')
            self.pi_optimize_expr = build_optimization(
                self.pi_loss + policy_decay, pi_lr, 'sac/pi')

            # for inference
            self.action = squashed_action_t[0]
            self.value = tf.reshape(v_t, [-1])[0]
            self.log_prob = log_prob_t[0]

    def _infer_arguments(self):
        return ['obs_t']

    def _update_arguments(self):
        return ['obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1']
