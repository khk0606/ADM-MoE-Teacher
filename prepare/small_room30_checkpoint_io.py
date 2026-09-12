"""Restricted tensor-checkpoint loading for old and new Torch, including Py3.8.

Never retry an unsafe/invalid file through unrestricted pickle. Old Torch has no
weights_only switch: provide a narrow pickle Unpickler that only rebuilds plain
tensor/Parameter storage and OrderedDict. Custom classes, NumPy pickles, tensor
subclasses and executable globals are deliberately unsupported. This is not a
sandbox against resource exhaustion; only use the intended local checkpoints.
"""
import inspect
import io
import pickle
import types
from collections import OrderedDict

import torch


class TensorCheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if module == 'torch._utils' and name in (
                '_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_parameter'):
            return getattr(torch._utils, name)
        if module == 'torch' and name in (
                'FloatStorage', 'DoubleStorage', 'HalfStorage', 'BFloat16Storage',
                'LongStorage', 'IntStorage', 'ShortStorage', 'CharStorage',
                'ByteStorage', 'BoolStorage', 'ComplexFloatStorage', 'ComplexDoubleStorage'):
            return getattr(torch, name)
        raise pickle.UnpicklingError('Blocked checkpoint global: %s.%s' % (module, name))


def _restricted_load(file, **kwargs):
    return TensorCheckpointUnpickler(file, **kwargs).load()


def _restricted_loads(data, **kwargs):
    return _restricted_load(io.BytesIO(data), **kwargs)


RESTRICTED_PICKLE = types.ModuleType('small_room30_restricted_pickle')
RESTRICTED_PICKLE.Unpickler = TensorCheckpointUnpickler
RESTRICTED_PICKLE.load = _restricted_load
RESTRICTED_PICKLE.loads = _restricted_loads


def load_checkpoint(path, map_location='cpu'):
    # Old torch.load exposes **pickle_load_args, NOT an explicit weights_only.
    # Signature detection avoids treating **kwargs as support for that keyword.
    if 'weights_only' in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=map_location, weights_only=True)
    return torch.load(path, map_location=map_location, pickle_module=RESTRICTED_PICKLE)
