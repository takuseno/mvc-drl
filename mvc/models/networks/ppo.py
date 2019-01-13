import tensorflow as tf

from mvc.models.networks.base_network import BaseNetwork
from mvc.action_output import ActionOutput


def build_value_loss(values, returns, value_factor):
    assert len(values.shape) == 2 and values.shape[1] == 1
    assert len(returns.shape) == 2 and returns.shape[1] == 1

    with tf.variable_scope('value_loss'):
        loss = tf.reduce_mean((returns - values) ** 2)
        return value_factor * loss

def build_entropy_loss(dist, entropy_factor):
    with tf.variable_scope('entropy'):
        entropy = -tf.reduce_mean(dist.entropy())
        return entropy_factor * entropy

def build_policy_loss(log_probs, old_log_probs, advantages, epsilon):
    assert len(log_probs.shape) == 2
    assert len(old_log_probs.shape) == 2
    assert old_log_probs.shape[1] == log_probs.shape[1]
    assert len(advantages.shape) == 2 and advantages.shape[1] == 1

    with tf.variable_scope('policy_loss'):
        ratio = tf.exp(log_probs - old_log_probs)
        ratio = tf.reduce_mean(ratio, axis=1, keepdims=True)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(
            ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
        surr = tf.minimum(surr1, surr2)
        loss = -tf.reduce_mean(surr)
    return loss

class PPONetwork(BaseNetwork):
    def __init__(self,
                 function,
                 state_shape,
                 num_envs,
                 num_actions,
                 batch_size,
                 epsilon,
                 lr,
                 grad_clip,
                 value_factor,
                 entropy_factor):

        self._build(function, state_shape, num_envs, num_actions, batch_size,
                    epsilon, lr, grad_clip, value_factor, entropy_factor)

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
            self.old_log_probs_ph: kwargs['log_probs_t']
        }
        sess = tf.get_default_session()
        return sess.run([self.loss, self.optimize_expr], feed_dict=feed_dict)[0]

    def _build(self,
               function,
               state_shape,
               num_envs,
               num_actions,
               batch_size,
               epsilon,
               lr,
               grad_clip,
               value_factor,
               entropy_factor):

        with tf.variable_scope('ppo', reuse=tf.AUTO_REUSE):
            # placeholers
            step_obs_ph = self.step_obs_ph = tf.placeholder(
                tf.float32, [num_envs] + list(state_shape), name='step_obs')
            train_obs_ph = self.train_obs_ph = tf.placeholder(
                tf.float32, [batch_size] + list(state_shape), name='train_obs')
            returns_ph = self.returns_ph = tf.placeholder(
                tf.float32, [batch_size], name='returns')
            advantages_ph = self.advantages_ph = tf.placeholder(
                tf.float32, [batch_size], name='advantages')
            actions_ph = self.actions_ph = tf.placeholder(
                tf.float32, [batch_size, num_actions], name='action')
            old_log_probs_ph = self.old_log_probs_ph = tf.placeholder(
                tf.float32, [batch_size, num_actions], name='old_log_prob')

            # network outputs for inference
            step_dist, step_values = function(step_obs_ph)
            # network outputs for training
            train_dist, train_values = function(train_obs_ph)

            # prepare for loss calculation
            advantages = tf.reshape(advantages_ph, [-1, 1])
            returns = tf.reshape(returns_ph, [-1, 1])
            log_probs = train_dist.log_prob(actions_ph)

            # individual loss
            value_loss = build_value_loss(train_values, returns, value_factor)
            entropy_loss = build_entropy_loss(train_dist, entropy_factor)
            policy_loss = build_policy_loss(log_probs, old_log_probs_ph,
                                            advantages, epsilon)
            # final loss
            self.loss = value_loss + policy_loss + entropy_loss

            # network weights
            network_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, 'ppo')

            # gradients
            gradients = tf.gradients(self.loss, network_vars)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
            # update
            grads_and_vars = zip(clipped_gradients, network_vars)
            optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-5)
            self.optimize_expr = optimizer.apply_gradients(grads_and_vars)

            # action
            self.action = step_dist.sample(1)[0]
            self.log_policy = step_dist.log_prob(self.action)
            self.value = tf.reshape(step_values, [-1])

    def _infer_arguments(self):
        return ['obs_t']

    def _update_arguments(self):
        return [
            'obs_t', 'actions_t', 'log_probs_t', 'returns_t', 'advantages_t'
        ]
