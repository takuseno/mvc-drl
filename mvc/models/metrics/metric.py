from mvc.models.metrics.base_metric import BaseMetric


class Metric(BaseMetric):
    def __init__(self):
        self.value = 0

    def add(self, value):
        self.value += value

    def get(self):
        return self.value

    def reset(self):
        self.value = 0
