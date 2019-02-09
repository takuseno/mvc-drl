import numpy as np


def make_batch(data, batch_size, data_size):
    assert isinstance(data, dict)

    indices = np.random.permutation(np.arange(data_size))
    for i in range(data_size // batch_size):
        index = indices[batch_size * i:batch_size * (i + 1)]
        batch = {}
        for key in data:
            batch[key] = data[key][index]
        yield batch
