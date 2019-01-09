import tensorflow as tf
import numpy as np

from unittest.mock import MagicMock
from mvc.parametric_function import _make_fcs
from mvc.parametric_function import stochastic_policy_function
from mvc.parametric_function import deterministic_policy_function


def make_inpt():
    dim1 = np.random.randint(10) + 1
    dim2 = np.random.randint(10) + 1
    return tf.constant(np.random.random((dim1, dim2)), dtype=tf.float32)

def make_fcs():
    return [np.random.randint(100) + 1 for i in range(np.random.randint(5) + 1)]

def mock_activation():
    return MagicMock(side_effect=tf.nn.relu)

def assert_hidden_variable_shape(variables, inpt, fcs):
    shapes = [(int(inpt.shape[1]), fcs[0])]
    shapes += shapes
    for i, fc in enumerate(fcs[1:]):
        shapes += [(fcs[i], fc)]
        shapes += [(fcs[i], fc)]
    for shape, variable in zip(shapes, variables):
        if str(variable.name).find('kernel') > 0:
            assert (int(variable.shape[0]), int(variable.shape[1])) == shape
        else:
            assert int(variable.shape[0]) == shape[1]

def assert_variable_range(variable, min_val, max_val):
    assert not np.any(variable < min_val)
    assert not np.any(variable > max_val)

def assert_variable_mismatch(variables1, variables2):
    for variable1, variable2 in zip(variables1, variables2):
        assert not np.all(variable1 == variable2)

class MakeFcsTest(tf.test.TestCase):
    def test_make_fcs(self):
        inpt = make_inpt()
        fcs = make_fcs()
        activation = mock_activation()
        w_init = tf.random_uniform_initializer(-0.1, 0.1)
        out = _make_fcs(fcs, inpt, activation, w_init)

        # to check connection
        optimizer = tf.train.AdamOptimizer(1e-4)
        optimize_expr = optimizer.minimize(tf.reduce_mean(out))

        # check variable shapes
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'hiddens')
        assert_hidden_variable_shape(variables, inpt, fcs)
        # check if activation is actually called
        assert activation.call_count == len(fcs)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            before = sess.run(variables)
            for variable in before:
                assert_variable_range(variable, -0.1, 0.1)

            sess.run(optimize_expr)

            after = sess.run(variables)
            assert_variable_mismatch(before, after)

class StochasticPolicyFunctionTest(tf.test.TestCase):
    def test_with_share_false(self):
        inpt = make_inpt()
        fcs = make_fcs()
        num_actions = np.random.randint(10) + 1
        w_init = tf.random_uniform_initializer(-0.1, 0.1)

        dist = stochastic_policy_function(
            fcs, inpt, num_actions, share=False,
            w_init=w_init, last_w_init=w_init)

        # to check connection
        optimizer = tf.train.AdamOptimizer(1e-4)
        optimize_expr = optimizer.minimize(tf.reduce_mean(dist.sample(1)))

        assert int(dist.sample(1)[0].shape[0]) == int(inpt.shape[0])
        assert int(dist.sample(1)[0].shape[1]) == num_actions 

        hiddens = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy/hiddens')
        assert_hidden_variable_shape(hiddens, inpt, fcs)

        mean = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy/mean')[0]
        assert int(mean.shape[0]) == fcs[-1]
        assert int(mean.shape[1]) == num_actions

        logstd = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy/logstd')[0]
        assert int(logstd.shape[0]) == 1
        assert int(logstd.shape[1]) == num_actions

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            before_mean = sess.run(mean)
            assert_variable_range(before_mean, -0.1, 0.1)

            before_logstd = sess.run(logstd)
            assert np.all(before_logstd == np.zeros_like(before_logstd))

            before = sess.run(hiddens)

            sess.run(optimize_expr)

            after_mean, after_logstd = sess.run([mean, logstd])
            assert_variable_mismatch([before_mean, before_logstd], [after_mean, after_logstd])

            after = sess.run(hiddens)
            assert_variable_mismatch(before, after)

    def test_with_share_true(self):
        inpt = make_inpt()
        fcs = make_fcs()
        num_actions = np.random.randint(10) + 1
        w_init = tf.random_uniform_initializer(-0.1, 0.1)

        dist = stochastic_policy_function(
            fcs, inpt, num_actions, share=True,
            w_init=w_init, last_w_init=w_init)

        assert int(dist.sample(1)[0].shape[0]) == int(inpt.shape[0])
        assert int(dist.sample(1)[0].shape[1]) == num_actions 

        logstd = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy/logstd')[0]
        assert int(logstd.shape[0]) == fcs[-1]
        assert int(logstd.shape[1]) == num_actions

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            before_logstd = sess.run(logstd)
            assert_variable_range(before_logstd, -0.1, 0.1)

class DeterministicPolicyFunctionTest(tf.test.TestCase):
    def test_deterministic_policy_function(self):
        inpt = make_inpt()
        fcs = make_fcs()
        num_actions = np.random.randint(10) + 1
        w_init = tf.random_uniform_initializer(-0.1, 0.1)

        policy = deterministic_policy_function(
            fcs, inpt, num_actions, w_init=w_init, last_w_init=w_init)

        # to check connection
        optimizer = tf.train.AdamOptimizer(1e-4)
        optimize_expr = optimizer.minimize(tf.reduce_mean(policy))

        assert int(policy.shape[0]) == int(inpt.shape[0])
        assert int(policy.shape[1]) == num_actions 

        hiddens = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy/hiddens')
        assert_hidden_variable_shape(hiddens, inpt, fcs)

        output = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy/output')[0]
        assert int(output.shape[0]) == fcs[-1]
        assert int(output.shape[1]) == num_actions

        variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            before = sess.run(variable)
            for var in before:
                assert_variable_range(var, -0.1, 0.1)

            sess.run(optimize_expr)

            after = sess.run(variable)
            assert_variable_mismatch(before, after)
