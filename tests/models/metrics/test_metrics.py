import numpy as np
import pytest

from unittest import TestCase
from unittest.mock import patch
from mvc.models.metrics.metrics import Metrics
from mvc.models.metrics.metric import Metric
from mvc.models.metrics.queue_metric import QueueMetric



class MetricsTest(TestCase):
    @patch('mvc.logger.set_adapter')
    @patch('mvc.logger.set_experiment_name')
    def test_init(self, set_experiment_name, set_adapter):
        metrics = Metrics('test1')
        set_experiment_name.assert_called_once_with('test1')

        metrics = Metrics('test2', 'visdom')
        set_adapter.assert_called_once_with('visdom', 'test2')

    @patch('mvc.logger.set_experiment_name')
    def test_register(self, set_experiment_name):
        metrics = Metrics('test')

        metrics.register('test1', 'single')
        assert isinstance(metrics.metrics['test1'], Metric)

        metrics.register('test2', 'queue')
        assert isinstance(metrics.metrics['test2'], QueueMetric)

        with pytest.raises(KeyError):
            metrics.register('test3', 'error')

        metrics.register('double', 'single')
        with pytest.raises(Exception):
            metrics.register('double', 'single')

    @patch('mvc.logger.set_experiment_name')
    def test_add_and_get(self, set_experiment_name):
        values = np.random.random(10)

        metrics = Metrics('test')
        metrics.register('test', 'single')
        for value in values:
            metrics.add('test', value)
        assert metrics.get('test') == np.sum(values)

        metrics.register('test2', 'queue')
        for value in values:
            metrics.add('test2', value)
        assert np.allclose(metrics.get('test2'), np.mean(values))

    
    @patch('mvc.logger.log_metric')
    @patch('mvc.logger.set_experiment_name')
    def test_log_metric(self, set_experiment_name, log_metric):
        metrics = Metrics('test')
        metrics.register('test1', 'single')

        value = np.random.random()
        metrics.add('test1', value)
        step = np.random.randint(10)
        metrics.log_metric('test1', step)

        log_metric.assert_called_once_with('test1', value, step)

    @patch('mvc.logger.log_parameters')
    @patch('mvc.logger.set_experiment_name')
    def test_log_parameters(self, set_experiment_name, log_parameters):
        metrics = Metrics('test')
        value = np.random.random()
        metrics.log_parameters({'test': value})

        log_parameters.assert_called_once_with({'test': value})
    
    @patch('mvc.logger.set_experiment_name')
    def test_reset(self, set_experiment_name):
        metrics = Metrics('test')
        metrics.register('test1', 'single')
        metrics.add('test1', np.random.random())
        metrics.reset('test1')

        assert metrics.get('test1') == 0

    @patch('mvc.logger.set_experiment_name')
    def test_has(self, set_experiment_name):
        metrics = Metrics('test')

        assert not metrics.has('test1')
        metrics.register('test1', 'single')
        assert metrics.has('test1')

        assert not metrics.has('test2')
        metrics.register('test2', 'queue')
        assert metrics.has('test2')

    @patch('mvc.logger.save_model')
    @patch('mvc.logger.set_experiment_name')
    def test_save_model(self, experiment_name, save_model):
        step = np.random.randint(10) + 1

        metrics = Metrics('test')
        metrics.save_model(step)
        save_model.assert_not_called()

        metrics = Metrics('test', saver='saver')
        metrics.save_model(step)
        save_model.assert_called_once_with('saver', step)
