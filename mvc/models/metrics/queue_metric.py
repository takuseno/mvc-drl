from collections import deque

import numpy as np

from mvc.models.metrics.base_metric import BaseMetric


class QueueMetric(BaseMetric):
    def __init__(self, maxlen=100):
        self.values = deque(maxlen=maxlen)

    def get(self):
        if self.values:
            return np.mean(list(self.values))
        return 0.0

    def add(self, value):
        self.values.append(value)

    def reset(self):
        self.values.clear()
