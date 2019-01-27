import tensorflow as tf
import argparse
import gym


from mvc.envs.wrappers import MuJoCoWrapper
from mvc.controllers.ddpg import DDPGController
from mvc.controllers.eval import EvalController
from mvc.models.networks.ddpg import DDPGNetwork
from mvc.models.metrics import Metrics
from mvc.models.buffer import Buffer
from mvc.view import View
from mvc.interaction import interact


def main(args):
    # environment
    env = MuJoCoWrapper(gym.make(args.env))
    eval_env = MuJoCoWrapper(gym.make(args.env))
    num_actions = env.action_space.shape[0]

    # deep neural network
    network = DDPGNetwork(args.layers, args.concat_index,
                          env.observation_space.shape, num_actions, args.gamma,
                          args.tau, args.actor_lr, args.critic_lr)

    # replay buffer
    buffer = Buffer(args.buffer_size)

    # metrics
    saver = tf.train.Saver()
    metrics = Metrics(args.name, args.log_adapter, saver)

    # controller
    controller = DDPGController(network, buffer, metrics, num_actions,
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
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 64],
                        help='layer units')
    parser.add_argument('--concat-index', type=int, default=1,
                        help='index of layer to concat action at q function')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='soft target update factor')
    parser.add_argument('--actor-lr', type=float, default=1e-4,
                        help='learning rate for actor')
    parser.add_argument('--critic-lr', type=float, default=1e-3,
                        help='learning rate for critic')
    parser.add_argument('--buffer-size', type=int, default=10 ** 6,
                        help='size of replay buffer')
    parser.add_argument('--batch-size', type=int, default=64,
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
    parser.add_argument('--log-adapter', type=str, help='(visdom, tfboard)')
    parser.add_argument('--load', type=str, help='path to model file')
    parser.add_argument('--render', action='store_true',
                        help='show rendered frames')
    args = parser.parse_args()
    main(args)
