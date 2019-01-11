import json

from visdom import Visdom
from mvc.logger.base_adapter import BaseAdapter


def json_to_html(json_str):
    html = '<div>'
    for line in json_str.split('\n'):
        html += '<p><span>' + line.replace(' ', '&nbsp;') + '</span></p>'
    html += '</div>'
    return html

class VisdomAdapter(BaseAdapter):
    def __init__(self, host, environment, experiment_name, port=8097):
        self.name = experiment_name
        self.visdom = Visdom(server=host, port=port, env=environment)

    def log_parameters(self, hyper_params):
        hyper_params['experiment_name'] = self.name
        html = json_to_html(json.dumps(hyper_params, indent=2))
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
