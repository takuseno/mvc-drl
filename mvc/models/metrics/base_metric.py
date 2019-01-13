class BaseMetric:
    def add(self, value):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()
