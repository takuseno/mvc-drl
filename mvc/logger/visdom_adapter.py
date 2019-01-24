import socket

from visdom import Visdom
from mvc.logger.base_adapter import BaseAdapter


def dict_to_html(params):
    html = '<table class="table table-bordered">'
    for key, value in params.items():
        html += '<tr><th>{}</th><td>{}</td></tr>'.format(key, value)
    html += '</table>'
    return html


class VisdomAdapter(BaseAdapter):
    def __init__(self, host, environment, experiment_name, port=8097):
        self.name = experiment_name
        self.visdom = Visdom(server=host, port=port, env=environment)

    def log_parameters(self, hyper_params):
        hyper_params['experiment_name'] = self.name
        hyper_params['host'] = socket.gethostname()
        html = dict_to_html(hyper_params)
        self.visdom.text(html)

    def set_model_graph(self, graph):
        pass

    def log_metric(self, name, metric, step):
        opts = {
            'showlegend': True,
            'title': name
        }
        self.visdom.line([metric], [step], name=self.name,
                         update='append', win=name, opts=opts)

    def register(self, name):
        pass
