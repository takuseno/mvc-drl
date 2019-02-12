import numpy as np

from mvc.misc.assertion import assert_type, assert_shape_match


def _assert_inputs(rewards, terminals, bootstrap_value):
    assert_type(rewards, np.ndarray)
    assert_type(terminals, np.ndarray)
    assert_shape_match(terminals, rewards)
    if len(rewards.shape) == 2:
        assert_type(bootstrap_value, np.ndarray)
        assert bootstrap_value.shape[0] == rewards.shape[1]


def compute_returns(bootstrap_value, rewards, terminals, gamma):
    _assert_inputs(rewards, terminals, bootstrap_value)

    returns = []
    return_tp1 = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        return_t = rewards[i] + (1.0 - terminals[i]) * gamma * return_tp1
        returns.append(return_t)
        return_tp1 = return_t
    returns = reversed(returns)
    return np.array(list(returns))


def compute_gae(bootstrap_value, rewards, values, terminals, gamma, lam):
    _assert_inputs(rewards, terminals, bootstrap_value)
    assert_type(values, np.ndarray)
    assert_shape_match(rewards, values)

    values = np.concatenate((values, [bootstrap_value]), axis=0)
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        return_t = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = return_t - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    adv_tp1 = deltas[-1]
    advantages = [adv_tp1]
    for i in reversed(range(deltas.shape[0] - 1)):
        adv_t = deltas[i] + (1.0 - terminals[i]) * gamma * lam * adv_tp1
        advantages.append(adv_t)
        adv_tp1 = adv_t
    advantages = reversed(advantages)
    return np.array(list(advantages))
