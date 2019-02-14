import tensorflow as tf
import numpy as np

from tests.test_utils import assert_variable_mismatch, assert_variable_match
from tests.test_utils import make_fcs, to_tf
from mvc.models.networks.sac import SACNetwork, SACNetworkParams
from mvc.models.networks.sac import build_v_loss
from mvc.models.networks.sac import build_q_loss
from mvc.models.networks.sac import build_pi_loss
from mvc.models.networks.sac import build_weight_decay


class BuildVLossTest(tf.test.TestCase):
    def test_success(self):
        nd_v_t = np.random.random((4, 1))
        nd_q1_t = np.random.random((4, 1))
        nd_q2_t = np.random.random((4, 1))
        nd_log_prob_t = np.random.random((4, 1))
        v_t = to_tf(nd_v_t)
        q1_t = to_tf(nd_q1_t)
        q2_t = to_tf(nd_q2_t)
        log_prob_t = to_tf(nd_log_prob_t)

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
        q_t = to_tf(nd_q_t)
        rewards_tp1 = to_tf(nd_rewards_tp1)
        v_tp1 = to_tf(nd_v_tp1)
        dones_tp1 = to_tf(nd_dones_tp1)

        loss = build_q_loss(q_t, rewards_tp1, v_tp1, dones_tp1, gamma)

        with self.test_session() as sess:
            target = nd_rewards_tp1 + gamma * nd_v_tp1 * (1.0 - nd_dones_tp1)
            answer = 0.5 * np.mean((target - nd_q_t) ** 2)
            assert np.allclose(sess.run(loss), answer)


class BuildPiLossTest(tf.test.TestCase):
    def test_success(self):
        nd_q1_t = np.random.random((4, 1))
        nd_q2_t = np.random.random((4, 1))
        nd_log_prob_t = np.random.random((4, 1))
        q1_t = to_tf(nd_q1_t)
        q2_t = to_tf(nd_q2_t)
        log_prob_t = to_tf(nd_log_prob_t)

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
        fcs = make_fcs()
        concat_index = np.random.randint(len(fcs))
        state_shape = (np.random.randint(5) + 1,)
        num_actions = np.random.randint(5) + 1
        gamma = np.random.random()
        tau = np.random.random()
        pi_lr = np.random.random()
        q_lr = np.random.random()
        v_lr = np.random.random()
        reg = np.random.random()
        self.params = SACNetworkParams(fcs=fcs, concat_index=concat_index,
                                       state_shape=state_shape,
                                       num_actions=num_actions,
                                       gamma=gamma, tau=tau, pi_lr=pi_lr,
                                       q_lr=q_lr, v_lr=v_lr, reg=reg)
        self.network = SACNetwork(self.params)

    def test_build(self):
        assert int(self.network.action.shape[0]) == self.params.num_actions
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
            obs = np.random.random(self.params.state_shape)
            output = self.network.infer(obs_t=obs)

        assert output.action.shape == (self.params.num_actions,)
        assert len(output.log_prob.shape) == 0
        assert len(output.value.shape) == 0

    def test_update(self):
        obs_t = np.random.random((32,) + self.params.state_shape)
        actions_t = np.random.random((32, self.params.num_actions))
        rewards_tp1 = np.random.random((32,))
        obs_tp1 = np.random.random((32,) + self.params.state_shape)
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
