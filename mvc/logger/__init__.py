import json
import csv
import logging

from mvc.logger.comet_ml_adapter import CometMlAdapter
from mvc.logger.visdom_adapter import VisdomAdapter


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

setting = {
    'path': None,
    'adapter': None,
    'verbose': True
}

def check_adapter():
    assert setting['adapter'] is not None

def set_adapter(adapter, experiment_name, config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.loads(f.read())
    if adapter == 'visdom':
        setting['adapter'] = VisdomAdapter(
            host=config['visdom']['host'], port=config['visdom']['port'],
            environment=config['visdom']['environment'],
            experiment_name=experiment_name)
    elif adapter == 'comet_ml':
        setting['adapter'] = CometMlAdapter(
            config['comet_ml']['api_key'], config['comet_ml']['project_name'],
            experiment_name)
    elif adapter == 'dummy':
        setting['adapter'] = DummyAdapter(**kwargs)
    else:
        raise KeyError()

def set_verbose(verbose):
    setting[verbose] = verbose

def log_parameters(hyper_params):
    assert isinstance(hyper_params, dict)
    check_adapter()
    setting['adapter'].log_parameters(hyper_params)
    if setting['verbose']:
        for key, value in hyper_params.items():
            logger.debug('{}={}'.format(key, value))

def set_model_graph(graph):
    check_adapter()
    setting['adapter'].set_model_path(graph)

def log_metric(name, metric, step):
    check_adapter()
    setting['adapter'].log_metric(name, metric, step)
    if setting['verbose']:
        logger.debug('step={} {}={}'.format(step, name, metric))
