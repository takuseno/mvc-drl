from unittest import TestCase
from mvc.models.metrics.metric import Metric


class MetricTest(TestCase):
    def test(self):
        metric = Metric()
        metric.add(1)
        assert metric.get() == 1
        metric.add(2)
        assert metric.get() == 3
        metric.reset()
        assert metric.get() == 0
