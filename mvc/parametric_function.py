import tensorflow as tf


def _make_fcs(fcs, inpt, activation=tf.nn.relu, w_init=None):
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
                               share=False,
                               w_init=None,
                               last_w_init=None):
    with tf.variable_scope('policy'):
        out = _make_fcs(fcs, inpt, tf.nn.tanh, w_init)
        mean = tf.layers.dense(out, num_actions, activation=None,
                               kernel_initializer=last_w_init, name='mean')

        if share:
            logstd = tf.layers.dense(out, num_actions, activation=None,
                                     kernel_initializer=last_w_init,
                                     name='logstd')
            std = tf.exp(logstd)
        else:
            logstd = tf.get_variable(name='logstd', shape=[1, num_actions],
                                     initializer=tf.zeros_initializer())
            std = tf.zeros_like(mean) + tf.exp(logstd)

        dist = tf.distributions.Normal(loc=mean, scale=std)
    return dist

def deterministic_policy_function(fcs,
                                  inpt,
                                  num_actions,
                                  w_init=None,
                                  last_w_init=None):
    with tf.variable_scope('policy'):
        out = _make_fcs(fcs, inpt, tf.nn.tanh, w_init)
        policy = tf.layers.dense(out, num_actions, activation=None,
                                 kernel_initializer=last_w_init,
                                 name='output')
    return policy

def value_function(fcs, inpt, w_init=None, last_w_init=None):
    with tf.variable_scope('value'):
        out = _make_fcs(fcs, inpt, tf.nn.tanh, w_init)
        value = tf.layers.dense(out, 1, activation=None,
                                kernel_initializer=last_w_init,
                                name='output')
    return value

def stochastic_function(fcs, num_actions, scope, w_init=None):
    def func(inpt):
        def last_w_init(scale):
            input_dim = int(inpt.shape[1])
            stddev = np.sqrt(scale / input_dim)
            return tf.random_normal_initializer(stddev=stddev)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            policy = stochastic_policy_function(fcs, inpt, num_actions,
                                                w_init, last_w_init(0.01))
            value = value_function(fcs, inpt, w_init, last_w_init(1.0))
        return policy, value
    return func

def deterministic_function(fcs, num_actions, scope, w_init=None):
    def func(inpt):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            policy = stochastic_policy_function(
                fcs, inpt, num_actions, w_init,
                tf.random_uniform_initializer(-3e-3, 3e-3))
            value = value_function(
                fcs, inpt, w_init, tf.random_uniform_initializer(-3e-4, 3e-4))
        return policy, value
    return func
