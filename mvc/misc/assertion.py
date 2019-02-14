def assert_type(variable, dtype):
    assert isinstance(variable, dtype), 'variable has an invalid type'


def assert_shape(array, shape):
    assert array.shape == shape, 'array has an invalid shape'


def assert_batch_shape(array, shape):
    assert_shape(array[0], shape)


def assert_batch_size_match(array1, array2):
    assert array1.shape[0] == array2.shape[0], 'invalid batch size'


def assert_shape_match(array1, array2):
    assert array1.shape == array2.shape, 'invalid shape'


def assert_shape_length(array, length):
    assert len(array.shape) == length, 'invalid shape size'


def assert_scalar(array):
    assert len(array.shape) == 2 and array.shape[1] == 1,\
        'array is not a scalar'
