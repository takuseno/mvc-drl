import pytest
import unittest

from mvc.models.networks.base_network import BaseNetwork


class BaseNetworkTest(unittest.TestCase):
    def setUp(self):
        self.network = BaseNetwork()

    def test_infer(self):
        with pytest.raises(NotImplementedError):
            self.network._infer()

    def test_update(self):
        with pytest.raises(NotImplementedError):
            self.network._update()

    def test_infer_arguments(self):
        with pytest.raises(NotImplementedError):
            self.network._infer_arguments()

    def test_update_arguments(self):
        with pytest.raises(NotImplementedError):
            self.network._update_arguments()
