import numpy as np


def compute_returns(bootstrap_value, rewards, terminals, gamma):
    assert isinstance(rewards, np.ndarray), 'rewards must be ndarray'
    assert isinstance(terminals, np.ndarray), 'terminals must be ndarray'
    assert rewards.shape == terminals.shape,\
        'rewards and terminals must have identical shape'
    if len(rewards.shape) == 2:
        assert isinstance(bootstrap_value, np.ndarray),\
            'bootstrap_value must be ndarray'
        assert bootstrap_value.shape[0] == rewards.shape[1],\
            'bootstrap_value must have column length of rewards'

    returns = []
    return_tp1 = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        return_t = rewards[i] + (1.0 - terminals[i]) * gamma * return_tp1
        returns.append(return_t)
        return_tp1 = return_t
    returns = reversed(returns)
    return np.array(list(returns))


def compute_gae(bootstrap_value, rewards, values, terminals, gamma, lam):
    assert isinstance(rewards, np.ndarray), 'rewards must be ndarray'
    assert isinstance(values, np.ndarray), 'values must be ndarray'
    assert isinstance(terminals, np.ndarray), 'terminals must be ndarray'
    assert rewards.shape == values.shape,\
        'rewards and values must have identical shape'
    assert rewards.shape == terminals.shape,\
        'rewards and terminals must have identical shape'
    if len(rewards.shape) == 2:
        assert isinstance(bootstrap_value, np.ndarray),\
            'bootstrap_value must be ndarray'
        assert bootstrap_value.shape[0] == rewards.shape[1],\
            'bootstrap_value must have column length of rewards'

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
