import json
import csv
import logging
import os

from datetime import datetime
from mvc.logger.visdom_adapter import VisdomAdapter
# from mvc.logger.comet_ml_adapter import CometMlAdapter


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

setting = {
    'path': 'logs',
    'adapter': None,
    'verbose': True,
    'writers': {},
    'experiment_name': None,
    'disable': False
}

def _write_csv(name, metric, step):
    if name not in setting['writers']:
        directory = os.path.join(setting['path'], setting['experiment_name'])
        path = os.path.join(directory, name + '.csv')
        if not os.path.exists(directory):
            os.makedirs(directory)
        file = open(path, 'w')
        setting['writers'][name] = csv.writer(file, lineterminator='\n')
    setting['writers'][name].writerow([step, metric])

def _write_hyper_params(parameters):
    directory = os.path.join(setting['path'], setting['experiment_name'])
    path = os.path.join(directory, 'hyper_params.csv')
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w') as f:
        f.write(json.dumps(parameters, indent=2))

def _load_config(path):
    with open(path, 'r') as f:
        return json.loads(f.read())

def set_adapter(adapter, experiment_name, config_path='config.json'):
    config = _load_config(config_path)

    set_experiment_name(experiment_name)

    if adapter == 'visdom':
        setting['adapter'] = VisdomAdapter(
            host=config['visdom']['host'], port=config['visdom']['port'],
            environment=config['visdom']['environment'],
            experiment_name=experiment_name)
    elif adapter == 'comet_ml':
        # setting['adapter'] = CometMlAdapter(
        #     config['comet_ml']['api_key'], config['comet_ml']['project_name'],
        #     experiment_name)
        assert KeyError('comet ml is not supported for this version')
    else:
        raise KeyError()

def set_experiment_name(experiment_name):
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    setting['experiment_name'] = experiment_name + '_' + date

def set_verbose(verbose):
    setting[verbose] = verbose

def enable():
    setting['disable'] = False

def disable():
    setting['disable'] = True

def log_parameters(hyper_params):
    assert isinstance(hyper_params, dict)
    assert setting['experiment_name'] is not None
    if setting['disable']:
        return

    if setting['adapter'] is not None:
        setting['adapter'].log_parameters(hyper_params)

    if setting['verbose']:
        for key, value in hyper_params.items():
            logger.debug('{}={}'.format(key, value))

    _write_hyper_params(hyper_params)

def set_model_graph(graph):
    if setting['disable']:
        return
    if setting['adapter'] is not None:
        setting['adapter'].set_model_path(graph)

def log_metric(name, metric, step):
    assert isinstance(name, str)
    assert isinstance(metric, int) or isinstance(metric, float)
    assert isinstance(step, int)
    assert setting['experiment_name'] is not None
    if setting['disable']:
        return

    if setting['adapter'] is not None:
        setting['adapter'].log_metric(name, metric, step)

    if setting['verbose']:
        logger.debug('step={} {}={}'.format(step, name, metric))

    _write_csv(name, metric, step)
