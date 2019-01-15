from mvc.action_output import ActionOutput


class BaseNetwork:
    def infer(self, **kwargs):
        for key in self._infer_arguments():
            assert key in kwargs, key + ' does not exist in the arguments'
        output = self._infer(**kwargs)

        assert isinstance(output, ActionOutput)

        return output

    def update(self, **kwargs):
        for key in self._update_arguments():
            assert key in kwargs, key + ' does not exist in the arguments'
        return self._update(**kwargs)

    def _infer(self, **kwargs):
        raise NotImplementedError()

    def _update(self, **kwargs):
        raise NotImplementedError()

    def _infer_arguments(self):
        raise NotImplementedError()

    def _update_arguments(self):
        raise NotImplementedError()
