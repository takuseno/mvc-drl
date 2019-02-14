import numpy as np
import pytest
import unittest

from mvc.models.metrics.base_metric import BaseMetric


class BaseMetricTest(unittest.TestCase):
    def setUp(self):
        self.metric = BaseMetric()

    def test_add(self):
        with pytest.raises(NotImplementedError):
            self.metric.add(np.random.random())

    def test_get(self):
        with pytest.raises(NotImplementedError):
            self.metric.get()

    def test_reset(self):
        with pytest.raises(NotImplementedError):
            self.metric.reset()
