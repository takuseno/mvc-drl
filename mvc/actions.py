class Action:
    def __init__(self, action, log_prob, value):
        self.action = action
        self.log_prob = log_prob
        self.value = value
