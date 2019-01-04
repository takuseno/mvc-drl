import tensorflow as tf
import tensorflow.layers as layers


def make_fcs(fcs, inpt, activation=tf.nn.relu, w_init=None):
    if w_init is None:
        w_init = tf.orthogonal_initializer(np.sqrt(2.0))
    out = inpt
    with tf.variable_scope('hiddens'):
        for hidden in fcs:
            out = layers.fully_connected(out, hidden, activation_fn=activation,
                                         weights_initializer=w_init)
    return out

def mlp_network(fcs,
                inpt,
                num_actions,
                nenvs,
                step_size,
                scope):
    input_dim = inpt.get_shape().as_list()[1] + 1
    def w_init(scale):
        return tf.random_normal_initializer(stddev=np.sqrt(scale / input_dim))

    def func(inpt):
        with tf.variable_scope('policy'):
            out = make_fcs(fcs, inpt, activation=tf.nn.tanh,
                           initializer=initializer(1.0))

            policy = layers.fully_connected(out, num_actions, activation_fn=None,
                                            weights_initializer=w_init(0.01))
            logstd = tf.get_variable(name='logstd', shape=[1, num_actions],
                                     initializer=tf.zeros_initializer())

            std = tf.zeros_like(policy) + tf.exp(logstd)
            dist = tf.distributions.Normal(loc=policy, scale=std)

        with tf.variable_scope('value'):
            out = make_fcs(fcs, inpt, activation=tf.nn.tanh,
                           initializer=initializer(1.0))
            value = layers.fully_connected(out, 1, activation_fn=None,
                                           weights_initializer=w_init(1.0))

        return dist, value

    return func
