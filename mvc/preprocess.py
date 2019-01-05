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
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
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
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    return np.array(list(advantages))
