from comet_ml import Experiment
from mvc.logger.base_adapter import BaseAdapter


class CometMlAdapter(BaseAdapter):
    def __init__(self, api_key, project_name, experiment_name):
        self.experiment = Experiment(api_key=api_key,
                                     project_name=project_name)
        self.experiment.set_name(experiment_name)

    def log_parameters(self, hyper_params):
        self.experiment.log_parameters(hyper_params)

    def set_model_graph(self, graph):
        self.experiment.set_model_graph(graph)

    def log_metric(self, name, metric, step):
        self.experiment.log_metric(name, metric, step=step)

    def register(self, name):
        pass
