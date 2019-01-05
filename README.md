[![CircleCI](https://circleci.com/gh/takuseno/well-tested-drl.svg?style=svg&circle-token=a53a3796ed3591f9f3bd411807367df4a23483c0)](https://circleci.com/gh/takuseno/well-tested-drl)

# mvc-drl
Clean deep reinforcement learning codes based on Web MVC architecture with complete unit tests

## motivation
Implementing deep reinforcement learning algorithms is easy to make up messy codes because interaction loop between an environment and an agent requires a lot of dependencies among classes.
Even deep learning requires special skills to build clean codes.

To think out of the box, Web engineers spent years on studying MVC (model-view-controller) architecture to build systems with tidy codes to handle interaction between Web and users.
Here, I found that this MVC architecture is very useful insight even for deep reinforcement learning implementation.
MVC provides a direction to an architecture with less dependencies, which would be nicer for unit testing.


## algorithms
First, this repository offers Proximal Policy Optimization (PPO) algorithm because PPO requires advantage computation and a bit complex neural network loss functions, which would be enough difficult for starters to implement.

- [x] Proximal Policy Optimization
- [ ] Deep Deterministic Policy Gradients
- [ ] Soft Actor-Critic

## unit testing
To gurantee code quality, all functions and classes including neural networks must have unit tests.

Following command runs all unit tests under `tests` directory.
```sh
$ ./test.sh
```