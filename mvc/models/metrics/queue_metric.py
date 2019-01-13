import numpy as np

from collections import deque
from mvc.models.metrics.base_metric import BaseMetric


class QueueMetric(BaseMetric):
    def __init__(self, maxlen=100):
        self.values = deque(maxlen=maxlen)
    
    def get(self):
        if len(self.values) > 0:
            return np.mean(list(self.values))
        return 0.0

    def add(self, value):
        self.values.append(value)

    def reset(self):
        self.values.clear()
