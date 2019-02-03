[![CircleCI](https://circleci.com/gh/takuseno/mvc-drl.svg?style=svg&circle-token=a53a3796ed3591f9f3bd411807367df4a23483c0)](https://circleci.com/gh/takuseno/mvc-drl)

# mvc-drl
Clean deep reinforcement learning codes based on Web MVC architecture with complete unit tests

## motivation
Implementing deep reinforcement learning algorithms is easy to make up messy codes because interaction loop between an environment and an agent requires a lot of dependencies among classes.
Even deep learning requires special skills to build clean codes.

To think out of the box, Web engineers spent years on studying MVC (model-view-controller) architecture to build systems with tidy codes to handle interaction between Web and users.
Here, I found that this MVC architecture is very useful insight even for deep reinforcement learning implementation.
MVC provides a direction to an architecture with less dependencies, which would be nicer for unit testing.


## algorithms
For academic usage, we provide baseline implementations that you might need to compare.

- [x] Proximal Policy Optimization
- [x] Deep Deterministic Policy Gradients
- [x] Soft Actor-Critic

## Ant performance
Each point represents an average evaluation reward of 10 episodes.

### PPO
![ppo](graphs/ppo_ant.png)

### DDPG
coming soon

### SAC
coming soon

## unit testing
To gurantee code quality, all functions and classes including neural networks must have unit tests.

Following command runs all unit tests under `tests` directory.
```sh
$ ./test.sh
```
