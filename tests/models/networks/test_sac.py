import tensorflow as tf
import numpy as np

from tests.test_utils import assert_variable_mismatch, assert_variable_match
from tests.test_utils import make_fcs
from mvc.models.networks.sac import SACNetwork
from mvc.models.networks.sac import build_v_loss
from mvc.models.networks.sac import build_q_loss
from mvc.models.networks.sac import build_pi_loss
from mvc.models.networks.sac import build_weight_decay


class BuildVLossTest(tf.test.TestCase):
    def test_success(self):
        nd_v_t = np.random.random((4, 1))
        nd_q1_t = np.random.random((4, 1))
        nd_q2_t = np.random.random((4, 1))
        nd_log_prob_t = np.random.random((4, 5))
        v_t = tf.constant(nd_v_t, dtype=tf.float32)
        q1_t = tf.constant(nd_q1_t, dtype=tf.float32)
        q2_t = tf.constant(nd_q2_t, dtype=tf.float32)
        log_prob_t = tf.constant(nd_log_prob_t, dtype=tf.float32)
        
        loss = build_v_loss(v_t, q1_t, q2_t, log_prob_t)

        with self.test_session() as sess:
            nd_q_t = np.minimum(nd_q1_t, nd_q2_t)
            answer = 0.5 * np.mean((nd_v_t - (nd_q_t - nd_log_prob_t)) ** 2)
            assert np.allclose(sess.run(loss), answer)


class BuildQLossTest(tf.test.TestCase):
    def test_success(self):
        nd_q_t = np.random.random((4, 1))
        nd_rewards_tp1 = np.random.random((4, 1))
        nd_v_tp1 = np.random.random((4, 1))
        nd_dones_tp1 = np.random.randint(2, size=(4, 1))
        gamma = np.random.random()
        q_t = tf.constant(nd_q_t, dtype=tf.float32)
        rewards_tp1 = tf.constant(nd_rewards_tp1, dtype=tf.float32)
        v_tp1 = tf.constant(nd_v_tp1, dtype=tf.float32)
        dones_tp1 = tf.constant(nd_dones_tp1, dtype=tf.float32)

        loss = build_q_loss(q_t, rewards_tp1, v_tp1, dones_tp1, gamma)

        with self.test_session() as sess:
            target = nd_rewards_tp1 + gamma * nd_v_tp1 * (1.0 - nd_dones_tp1)
            answer = 0.5 * np.mean((target - nd_q_t) ** 2)
            assert np.allclose(sess.run(loss), answer)


class BuildPiLossTest(tf.test.TestCase):
    def test_success(self):
        nd_q1_t = np.random.random((4, 1))
        nd_q2_t = np.random.random((4, 1))
        nd_log_prob_t = np.random.random((4, 5))
        q1_t = tf.constant(nd_q1_t, dtype=tf.float32)
        q2_t = tf.constant(nd_q2_t, dtype=tf.float32)
        log_prob_t = tf.constant(nd_log_prob_t, dtype=tf.float32)

        loss = build_pi_loss(log_prob_t, q1_t, q2_t)

        with self.test_session() as sess:
            q_t = np.minimum(nd_q1_t, nd_q2_t)
            answer = np.mean(nd_log_prob_t - q_t)
            assert np.allclose(sess.run(loss), answer)


class BuildWeightDecayTest(tf.test.TestCase):
    def test_success(self):
        tf.reset_default_graph()
        nd_weight = np.random.random((4, 4))
        nd_bias = np.random.random((1, 4))
        weight = tf.Variable(nd_weight, name='var/weight')
        bias = tf.Variable(nd_bias, name='var/bias')
        scale = np.random.random()

        weight_decay = build_weight_decay(scale, 'var')
        weight_sum = tf.reduce_sum(weight)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            assert np.allclose(sess.run(weight_decay), scale * sess.run(weight_sum))


class SACNetworkTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.fcs = make_fcs()
        self.concat_index = np.random.randint(len(self.fcs))
        self.state_shape = (np.random.randint(5) + 1,)
        self.num_actions = np.random.randint(5) + 1
        self.gamma = np.random.random()
        self.tau = np.random.random()
        self.pi_lr = np.random.random()
        self.q_lr = np.random.random()
        self.v_lr = np.random.random()
        self.network = SACNetwork(self.fcs, self.concat_index,
                                  self.state_shape, self.num_actions,
                                  self.gamma, self.tau, self.pi_lr,
                                  self.q_lr, self.v_lr)
    def test_build(self):
        assert int(self.network.action.shape[0]) == self.num_actions
        assert len(self.network.value.shape) == 0
        assert len(self.network.pi_loss.shape) == 0
        assert len(self.network.v_loss.shape) == 0
        assert len(self.network.q1_loss.shape) == 0
        assert len(self.network.q2_loss.shape) == 0

    def test_infer_arguments(self):
        args = self.network._infer_arguments()
        keys = ['obs_t']
        for key in args:
            assert key in args

    def test_update_arguments(self):
        args = self.network._update_arguments()
        keys = ['obs_t', 'actions_t', 'rewards_tp1', 'obs_tp1', 'dones_tp1']
        for key in args:
            assert key in keys

    def test_infer(self):
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            obs = np.random.random(self.state_shape)
            output = self.network.infer(obs_t=obs)

        assert output.action.shape == (self.num_actions,)
        assert output.log_prob.shape == (self.num_actions,)
        assert len(output.value.shape) == 0

    def test_update(self):
        obs_t = np.random.random((32,) + self.state_shape)
        actions_t = np.random.random((32, self.num_actions))
        rewards_tp1 = np.random.random((32,))
        obs_tp1 = np.random.random((32,) + self.state_shape)
        dones_tp1 = np.random.random((32,))
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'sac')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(variables)

            v_loss, (q1_loss, q2_loss), pi_loss = self.network.update(
                obs_t=obs_t, actions_t=actions_t, rewards_tp1=rewards_tp1,
                obs_tp1=obs_tp1, dones_tp1=dones_tp1)

            after = sess.run(variables)

        assert_variable_mismatch(before, after)
