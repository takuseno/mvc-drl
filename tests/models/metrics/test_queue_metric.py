import numpy as np

from unittest import TestCase
from mvc.models.metrics.queue_metric import QueueMetric


class QueueMetricTest(TestCase):
    def test(self):
        values = np.random.random(10)
        metric = QueueMetric()
        for i in range(10):
            metric.add(values[i])
            assert np.allclose(metric.get(), np.mean(values[:i + 1]))
        metric.reset()
        assert metric.get() == 0.0

    def test_capacity(self):
        values = np.random.random(10)
        metric = QueueMetric(maxlen=5)

        for i in range(10):
            metric.add(values[i])

        assert np.allclose(metric.get(), np.mean(values[5:]))
