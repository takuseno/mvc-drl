import tensorflow as tf

from mvc.logger.base_adapter import BaseAdapter


class TfBoardAdapter(BaseAdapter):
    def __init__(self, logdir):
        self.logdir = logdir
        self.placeholders = {}
        self.summaries = {}
        self.writer = None

    def log_parameters(self, hyper_params):
        pass

    def set_model_graph(self, graph):
        self.writer = tf.summary.FileWriter(self.logdir, graph)

    def log_metric(self, name, metric, step):
        assert self.writer is not None, 'call set_model_graph first'

        sess = tf.get_default_session()
        placeholder = self.placeholders[name]
        summary = self.summaries[name]
        feed_dict = {placeholder: metric}
        out, _ = sess.run([summary, placeholder], feed_dict=feed_dict)
        self.writer.add_summary(out, step)

    def register(self, name):
        placeholder = tf.placeholder(tf.float32, [], name=name)
        self.placeholders[name] = placeholder
        self.summaries[name] = tf.summary.scalar(name + '_smmry', placeholder)
