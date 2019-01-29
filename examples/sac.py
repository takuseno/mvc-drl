import tensorflow as tf
import numpy as np
import argparse
import gym


from mvc.envs.wrappers import MuJoCoWrapper
from mvc.controllers.sac import SACController
from mvc.controllers.eval import EvalController
from mvc.models.networks.sac import SACNetwork
from mvc.models.metrics import Metrics
from mvc.models.buffer import Buffer
from mvc.noise import EmptyNoise
from mvc.view import View
from mvc.interaction import interact


def main(args):
    # environment
    env = MuJoCoWrapper(gym.make(args.env), args.reward_scale, args.render)
    eval_env = MuJoCoWrapper(gym.make(args.env))
    num_actions = env.action_space.shape[0]

    # deep neural network
    network = SACNetwork(args.layers, args.concat_index,
                          env.observation_space.shape, num_actions, args.gamma,
                          args.tau, args.pi_lr, args.q_lr, args.v_lr)

    # replay buffer
    buffer = Buffer(args.buffer_size)

    # metrics
    saver = tf.train.Saver()
    metrics = Metrics(args.name, args.log_adapter, saver)

    # exploration noise
    noise = EmptyNoise()

    # controller
    controller = SACController(network, buffer, metrics, noise, num_actions,
                                args.batch_size, args.final_steps,
                                args.log_interval, args.save_interval,
                                args.eval_interval)

    # view
    view = View(controller)

    # evaluation
    eval_controller = EvalController(network, metrics, args.eval_episode)
    eval_view = View(eval_controller)

    # save hyperparameters
    metrics.log_parameters(vars(args))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # save model graph for debugging
        metrics.set_model_graph(sess.graph)

        if args.load is not None:
            saver.restore(sess, args.load)

        interact(env, view, eval_env, eval_view)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='training environment')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='reward scaling')
    parser.add_argument('--layers', type=int, nargs='+', default=[256, 256],
                        help='layer units')
    parser.add_argument('--concat-index', type=int, default=1,
                        help='index of layer to concat action at q function')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='soft target update factor')
    parser.add_argument('--pi-lr', type=float, default=3e-4,
                        help='learning rate for plicy function')
    parser.add_argument('--q-lr', type=float, default=3e-4,
                        help='learning rate for q functions')
    parser.add_argument('--v-lr', type=float, default=3e-4,
                        help='learning rate for value functions')
    parser.add_argument('--buffer-size', type=int, default=10 ** 6,
                        help='size of replay buffer')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--final-steps', type=int, default=10 ** 6,
                        help='the number of training steps')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='log interval')
    parser.add_argument('--save-interval', type=int, default=10 ** 5,
                        help='save interval')
    parser.add_argument('--eval-interval', type=int, default=10 ** 4,
                        help='evaluation interval')
    parser.add_argument('--eval-episode', type=int, default=10,
                        help='the number of evaluation episode')
    parser.add_argument('--name', type=str, default='sac',
                        help='name of experiment')
    parser.add_argument('--log-adapter', type=str, help='(visdom, tfboard)')
    parser.add_argument('--load', type=str, help='path to model file')
    parser.add_argument('--render', action='store_true',
                        help='show rendered frames')
    args = parser.parse_args()
    main(args)
