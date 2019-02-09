import numpy as np
import unittest
import pytest

from mvc.misc.batch import make_batch


class MakeBatchTest(unittest.TestCase):
    def test_success(self):
        batch_size = np.random.randint(32) + 1
        data_size = batch_size * np.random.randint(10) + 1
        data = {
            'test1': np.random.random((data_size)),
            'test2': np.random.random((data_size))
        }
        count = 0
        for batch in make_batch(data, batch_size, data_size):
            assert batch['test1'].shape[0] == batch_size
            assert batch['test2'].shape[0] == batch_size
            count += 1
        assert count == data_size // batch_size
    
    def test_assertion_error(self):
        with pytest.raises(AssertionError):
            data = [1, 2, 3]
            batch_size = np.random.randint(32) + 1
            data_size = np.random.randint(1024) + 1
            batch = make_batch(data, batch_size, data_size)
            next(batch)
