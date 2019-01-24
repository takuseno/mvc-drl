class BaseAdapter:
    def log_parameters(self, hyper_params):
        raise NotImplementedError()

    def set_model_graph(self, graph):
        raise NotImplementedError()

    def log_metric(self, name, metric, step):
        raise NotImplementedError()

    def register(self, name):
        raise NotImplementedError()
