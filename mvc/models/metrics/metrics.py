import mvc.logger as logger

from mvc.models.metrics.metric import Metric
from mvc.models.metrics.queue_metric import QueueMetric


class Metrics:
    def __init__(self, experiment_name, adapter=None):
        self.metrics = {}
        if adapter is None:
            logger.set_experiment_name(experiment_name)
        else:
            logger.set_adapter(adapter, experiment_name)

    def register(self, name, mode, **kwargs):
        if name in self.metrics:
            raise Exception(name + ' already exists')

        if mode == 'single':
            self.metrics[name] = Metric()
        elif mode == 'queue':
            self.metrics[name] = QueueMetric(**kwargs)
        else:
            raise KeyError()

    def add(self, name, value):
        self._check_name(name)
        self.metrics[name].add(value)

    def get(self, name):
        self._check_name(name)
        return self.metrics[name].get()

    def has(self, name):
        return name in self.metrics

    def log_metric(self, name, step):
        self._check_name(name)
        logger.log_metric(name, self.get(name), step)

    def log_parameters(self, hyper_params):
        logger.log_parameters(hyper_params)

    def reset(self, name):
        self._check_name(name)
        self.metrics[name].reset()

    def _check_name(self, name):
        assert name in self.metrics, name + ' must be registered'
