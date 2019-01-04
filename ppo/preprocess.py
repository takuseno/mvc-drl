def compute_returns(bootstrap_value, rewards, terminals, gamma):
    returns = []
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    return np.array(list(returns))

def compute_gae(bootstrap_value, rewards, values, terminals, gamma, lam):
    values = np.vstack([values, bootstrap_value])
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1,:]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    return np.array(list(advantages))
