import json
import csv
import logging
import os

from datetime import datetime

import tensorflow as tf

from mvc.logger.visdom_adapter import VisdomAdapter
from mvc.logger.tfboard_adapter import TfBoardAdapter
# from mvc.logger.comet_ml_adapter import CometMlAdapter


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

SETTING = {
    'path': 'logs',
    'adapter': None,
    'verbose': True,
    'writers': {},
    'experiment_name': None,
    'disable': False
}


def _get_dir():
    return os.path.join(SETTING['path'], SETTING['experiment_name'])


def _prepare_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _write_csv(name, metric, step):
    if name not in SETTING['writers']:
        directory = _get_dir()
        _prepare_dir(directory)
        path = os.path.join(directory, name + '.csv')
        file = open(path, 'w')
        SETTING['writers'][name] = csv.writer(file, lineterminator='\n')
    SETTING['writers'][name].writerow([step, metric])


def _write_hyper_params(parameters):
    directory = _get_dir()
    _prepare_dir(directory)
    path = os.path.join(directory, 'hyper_params.json')
    with open(path, 'w') as file:
        file.write(json.dumps(parameters, indent=2))


def _load_config(path):
    with open(path, 'r') as file:
        return json.loads(file.read())


def set_adapter(adapter, experiment_name, config_path='config.json'):
    set_experiment_name(experiment_name)

    if adapter == 'visdom':
        config = _load_config(config_path)
        SETTING['adapter'] = VisdomAdapter(
            host=config['visdom']['host'], port=config['visdom']['port'],
            environment=config['visdom']['environment'],
            experiment_name=SETTING['experiment_name'])
    elif adapter == 'tfboard':
        SETTING['adapter'] = TfBoardAdapter(_get_dir())
    elif adapter == 'comet_ml':
        assert KeyError('comet ml is not supported for this version')
    else:
        raise KeyError()


def set_experiment_name(experiment_name):
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    SETTING['experiment_name'] = experiment_name + '_' + date


def set_verbose(verbose):
    SETTING[verbose] = verbose


def enable():
    SETTING['disable'] = False


def disable():
    SETTING['disable'] = True


def log_parameters(hyper_params):
    assert isinstance(hyper_params, dict)
    assert SETTING['experiment_name'] is not None
    if SETTING['disable']:
        return

    if SETTING['adapter'] is not None:
        SETTING['adapter'].log_parameters(hyper_params)

    if SETTING['verbose']:
        for key, value in hyper_params.items():
            LOGGER.debug('%s=%s', key, value)

    _write_hyper_params(hyper_params)


def set_model_graph(graph):
    if SETTING['disable']:
        return
    if SETTING['adapter'] is not None:
        SETTING['adapter'].set_model_graph(graph)


def save_model(saver, step):
    if SETTING['disable']:
        return
    sess = tf.get_default_session()
    directory = _get_dir()
    _prepare_dir(directory)
    path = os.path.join(directory, 'model.ckpt')
    saver.save(sess, path, global_step=step)


def log_metric(name, metric, step):
    assert isinstance(name, str)
    assert isinstance(step, int)
    assert SETTING['experiment_name'] is not None
    if SETTING['disable']:
        return

    if SETTING['adapter'] is not None:
        SETTING['adapter'].log_metric(name, metric, step)

    if SETTING['verbose']:
        LOGGER.debug('step=%d %s=%f', step, name, metric)

    _write_csv(name, metric, step)


def register(name):
    if SETTING['adapter'] is not None:
        SETTING['adapter'].register(name)
