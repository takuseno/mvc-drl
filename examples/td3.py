import tensorflow as tf
import numpy as np
import argparse
import gym
import pybullet_envs
import roboschool


from mvc.envs.wrappers import MuJoCoWrapper
from mvc.controllers.td3 import TD3Controller
from mvc.controllers.eval import EvalController
from mvc.models.networks.td3 import TD3Network, TD3NetworkParams
from mvc.models.metrics import Metrics
from mvc.models.buffer import Buffer
from mvc.noise import NormalActionNoise
from mvc.view import View
from mvc.interaction import interact


def main(args):
    # environment
    env = MuJoCoWrapper(gym.make(args.env), args.reward_scale, args.render)
    env.seed(args.seed)
    eval_env = MuJoCoWrapper(gym.make(args.env))
    eval_env.seed(args.seed)
    num_actions = env.action_space.shape[0]

    # network parameters
    params = TD3NetworkParams(fcs=args.layers, concat_index=args.concat_index,
                               state_shape=env.observation_space.shape,
                               num_actions=num_actions, gamma=args.gamma,
                               tau=args.tau, actor_lr=args.actor_lr,
                               critic_lr=args.critic_lr,
                               target_noise_sigma=args.target_noise_sigma,
                               target_noise_clip=args.target_noise_clip)

    # deep neural network
    network = TD3Network(params)

    # replay buffer
    buffer = Buffer(args.buffer_size)

    # metrics
    saver = tf.train.Saver()
    metrics = Metrics(args.name, args.log_adapter, saver)

    # exploration noise
    noise = NormalActionNoise(np.zeros(num_actions), np.ones(num_actions) * 0.1)

    # controller
    controller = TD3Controller(network, buffer, metrics, noise, num_actions,
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
    parser.add_argument('--layers', type=int, nargs='+', default=[400, 300],
                        help='layer units')
    parser.add_argument('--concat-index', type=int, default=1,
                        help='index of layer to concat action at q function')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=5e-3,
                        help='soft target update factor')
    parser.add_argument('--actor-lr', type=float, default=1e-3,
                        help='learning rate for actor')
    parser.add_argument('--critic-lr', type=float, default=1e-3,
                        help='learning rate for critic')
    parser.add_argument('--buffer-size', type=int, default=10 ** 6,
                        help='size of replay buffer')
    parser.add_argument('--batch-size', type=int, default=100,
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
    parser.add_argument('--name', type=str, default='ddpg',
                        help='name of experiment')
    parser.add_argument('--log-adapter', type=str, default='tfboard',
                        help='(visdom, tfboard)')
    parser.add_argument('--load', type=str, help='path to model file')
    parser.add_argument('--render', action='store_true',
                        help='show rendered frames')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of environment')
    parser.add_argument('--target-noise-sigma', type=float, default=0.2,
                        help='Standard deviation of target policy noise')
    parser.add_argument('--target-noise-clip', type=float, default=0.5,
                        help='Clip value of target policy')
    args = parser.parse_args()
    main(args)
