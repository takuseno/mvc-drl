[![CircleCI](https://circleci.com/gh/takuseno/mvc-drl.svg?style=svg&circle-token=a53a3796ed3591f9f3bd411807367df4a23483c0)](https://circleci.com/gh/takuseno/mvc-drl)

# mvc-drl
Clean deep reinforcement learning codes based on Web MVC architecture with complete unit tests

## motivation
Implementing deep reinforcement learning algorithms is easy to make up messy codes because interaction loop between an environment and an agent requires a lot of dependencies among classes.
Even deep learning requires special skills to build clean codes.

To think out of the box, Web engineers spent years on studying MVC (model-view-controller) architecture to build systems with tidy codes to handle interaction between Web and users.
Here, I found that this MVC architecture is very useful insight even for deep reinforcement learning implementation.
MVC provides a direction to an architecture with less dependencies, which would be nicer for unit testing.

## installation
### nvidia-docker
You can use docker to setup and run experiments.
```
$ ./scripts/build.sh
```

Once you built the container, you can start a container with nvidia runtime via `./scripts/up.sh`.
```
$ ./scripts/up.sh
root@a84ab59aa668:/home/app#  ls
Dockerfile  README.md    example.confing.json  graphs            mvc      scripts  tests
LICENSE     examples     logs                  requirements.txt  test.sh  tools
root@a84ab59aa668:/home/app#  ls
```

### manual
You need to install packages written in `requirements.txt` and tensorflow.
```
$ pip install -r requirements.txt
$ pip install tensorflow-gpu>=1.12.0
```


## algorithms
For academic usage, we provide baseline implementations that you might need to compare.

- [x] Proximal Policy Optimization
- [x] Deep Deterministic Policy Gradients
- [x] Soft Actor-Critic

## Ant performance
Each point represents an average evaluation reward of 10 episodes.
Almost same performance has been achieved as a paper of [Soft Actor-Critic](https://arxiv.org/abs/1801.01290).

### PPO
```sh
$ python -m examples.ppo --env Ant-v2
```

![ppo](graphs/ppo_ant.png)

### DDPG
```sh
$ python -m examples.ddpg --env Ant-v2
```

![ddpg](graphs/ddpg_ant.png)

### SAC
coming soon

## unit testing
To gurantee code quality, all functions and classes including neural networks must have unit tests.

Following command runs all unit tests under `tests` directory.
```sh
$ ./test.sh
```
