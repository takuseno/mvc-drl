import tensorflow as tf
import argparse
import gym

from mvc.envs.wrappers import BatchEnvWrapper, MuJoCoWrapper
from mvc.controllers.ppo import PPOController
from mvc.controllers.eval import EvalController
from mvc.models.networks.ppo import PPONetwork
from mvc.models.metrics import Metrics
from mvc.models.rollout import Rollout
from mvc.view import View
from mvc.interaction import batch_interact
from mvc.parametric_function import stochastic_function


def make_envs(env_name, num_envs):
    return [MuJoCoWrapper(gym.make(env_name)) for _ in range(num_envs)]

def main(args):
    env = BatchEnvWrapper(
        make_envs(args.env, args.num_envs), render=args.render)
    eval_env = BatchEnvWrapper(make_envs(args.env, args.num_envs))

    num_actions = env.action_space.shape[0]

    function = stochastic_function(args.layers, num_actions, 'ppo')

    network = PPONetwork(function, env.observation_space.shape, args.num_envs,
                         num_actions, args.batch_size, args.epsilon,
                         args.lr, args.grad_clip, args.value_factor,
                         args.entropy_factor)

    rollout = Rollout()

    saver = tf.train.Saver()
    metrics = Metrics(args.name, args.log_adapter, saver)

    controller = PPOController(network, rollout, metrics, args.num_envs,
                               args.time_horizon, args.epoch, args.batch_size,
                               args.gamma, args.lam, args.final_steps,
                               args.log_interval,  args.save_interval,
                               args.eval_interval)
    view = View(controller)

    eval_controller = EvalController(network, metrics, args.eval_episodes)
    eval_view = View(eval_controller)

    # save hyperparameters
    metrics.log_parameters(vars(args))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # save model graph for debugging
        metrics.set_model_graph(sess.graph)

        if args.load is not None:
            saver.restore(sess, args.load)

        batch_interact(env, view, eval_env, eval_view)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time-horizon', type=int,
                        default=2048, help='interval to update')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='the number of environments')
    parser.add_argument('--epoch', type=int, default=10,
                        help='epoch of training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size of training')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='lambda of generalized advantage estimation')
    parser.add_argument('--log-interval', type=int, help='interval of logging')
    parser.add_argument('--final-steps', type=int, default=10 ** 6,
                        help='the number of training steps')
    parser.add_argument('--layers', type=int, nargs='+', default=[64, 64],
                        help='layer units')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='clipping factor')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--grad-clip', type=float, default=0.5,
                        help='gradient clipping')
    parser.add_argument('--value-factor', type=float, default=1.0,
                        help='value loss weight')
    parser.add_argument('--entropy-factor', type=float, default=0.0,
                        help='entropy loss weight')
    parser.add_argument('--env', type=str, default='Pendulum-v0',
                        help='training environment')
    parser.add_argument('--name', type=str, default='experiment',
                        help='experiment name')
    parser.add_argument('--log-adapter', type=str,
                        help='log adapter (visdom, comet_ml)')
    parser.add_argument('--save-interval', type=int, default=2048 * 50,
                        help='interval of saving parameters')
    parser.add_argument('--load', type=str, help='path to model')
    parser.add_argument('--eval-interval', type=int, default=2048 * 10,
                        help='interval of evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='the number of evaluation episode')
    parser.add_argument('--render', action='store_true',
                        help='show frames of environment')
    args = parser.parse_args()
    main(args)
