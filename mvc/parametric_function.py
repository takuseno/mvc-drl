import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def _make_fcs(fcs, inpt, activation, w_init=None):
    if w_init is None:
        w_init = tf.orthogonal_initializer(np.sqrt(2.0))
    out = inpt
    with tf.variable_scope('hiddens'):
        for i, hidden in enumerate(fcs):
            out = tf.layers.dense(out, hidden, activation=activation,
                                  kernel_initializer=w_init,
                                  name='hidden{}'.format(i))
    return out


def stochastic_policy_function(fcs,
                               inpt,
                               num_actions,
                               activation=tf.nn.tanh,
                               share=False,
                               w_init=None,
                               last_w_init=None,
                               last_b_init=None,
                               scope='policy'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = _make_fcs(fcs, inpt, activation, w_init)
        mean = tf.layers.dense(out, num_actions, activation=None,
                               kernel_initializer=last_w_init,
                               bias_initializer=last_b_init,
                               name='mean')

        if share:
            logstd = tf.layers.dense(out, num_actions, activation=None,
                                     kernel_initializer=last_w_init,
                                     bias_initializer=last_b_init,
                                     name='logstd')
            clipped_logstd = tf.clip_by_value(logstd, -20, 2)
            std = tf.exp(clipped_logstd)
        else:
            logstd = tf.get_variable(name='logstd', shape=[1, num_actions],
                                     initializer=tf.zeros_initializer())
            std = tf.zeros_like(mean) + tf.exp(logstd)

        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
    return dist


def deterministic_policy_function(fcs,
                                  inpt,
                                  num_actions,
                                  activation=tf.nn.tanh,
                                  w_init=None,
                                  last_w_init=None,
                                  last_b_init=None,
                                  scope='policy'):
    if last_b_init is None:
        last_b_init = tf.zeros_initializer()

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = _make_fcs(fcs, inpt, activation, w_init)
        policy = tf.layers.dense(out, num_actions, activation=None,
                                 kernel_initializer=last_w_init,
                                 bias_initializer=last_b_init,
                                 name='output')
    return policy


def value_function(fcs,
                   inpt,
                   activation=tf.nn.tanh,
                   w_init=None,
                   last_w_init=None,
                   last_b_init=None,
                   scope='value'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = _make_fcs(fcs, inpt, activation, w_init)
        value = tf.layers.dense(out, 1, activation=None,
                                kernel_initializer=last_w_init,
                                bias_initializer=last_b_init,
                                name='output')
    return value


def q_function(fcs,
               inpt,
               action,
               concat_index,
               activation=tf.nn.tanh,
               w_init=None,
               last_w_init=None,
               last_b_init=None,
               scope='action_value'):
    if last_b_init is None:
        last_b_init = tf.zeros_initializer()

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = inpt
        with tf.variable_scope('hiddens'):
            for i, hidden in enumerate(fcs):
                if i == concat_index:
                    out = tf.concat([out, action], axis=1)
                out = tf.layers.dense(out, hidden, activation=activation,
                                      kernel_initializer=w_init,
                                      name='hidden{}'.format(i))
        value = tf.layers.dense(out, 1, activation=None,
                                kernel_initializer=last_w_init,
                                bias_initializer=last_b_init,
                                name='output')
    return value
